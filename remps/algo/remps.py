"""
Relative entropy policy model search (REMPS)
Idea: use REMPS to find the distribution p(s,a,s') containing both policy and transition model.
Then matches the distributions minimizing the KL between the p and the induced distribution from
\pi and \p_\omega
Follows the rllab implementation
"""
from enum import Enum
from typing import Dict

import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
from baselines import logger
from tensorflow.contrib.opt import ScipyOptimizerInterface

from remps.envs.confmdp import ConfMDP
from remps.model_approx.model_approximator import ModelApproximator
from remps.policy.policy import Policy
from remps.utils.logging import timed
from remps.utils.utils import get_default_tf_dtype


class Projection(Enum):
    """
    Projection types.
    State kernel: joint projection
    DISJOINT: Project independently policy and model
    D_PROJECTION: Idea projection of the stationary distribution
    """
    STATE_KERNEL = 0
    DISJOINT = 1
    D_PROJECTION = 2


class REMPS:
    """
    Relative Entropy Model Policy Search (REMPS)
    """

    def __init__(
        self,
        kappa: float =1e-3,  # 0.001,
        L2_reg_dual: float=0.0,  # 1e-7,# 1e-5,
        L2_reg_loss: float=0.0,
        max_opt_itr: int=1000,
        tf_optimizer=ScipyOptimizerInterface,
        model: ModelApproximator=None,
        policy: Policy=None,
        env: ConfMDP =None,
        projection_type: Projection =Projection.STATE_KERNEL,  # State kernel or disjoint
        training_set_size: int=5000,
        exact: bool=False,
        restart_fitting: bool=False,
        fit_iterations: int=40000,
        refit_iterations: int=1000,
        refit: int=False,
        refit_every_iterations: int=100,
        **kwargs,
    ):
        """
        :param kappa: Max KL divergence between new policy and old policy.
        :param L2_reg_dual: Dual regularization
        :param L2_reg_loss: Loss regularization
        :param max_opt_itr: Maximum number of batch optimization iterations.
        :param tf_optimizer: optimizer to use
        :param model: model approximation
        :param policy: policy to be optimized
        :param env: environment
        :param projection_type: type of projection
        :param use_features: whether to use features or not
        :param training_set_size: number of samples in the training set
        :param exact: whether the model approximation is exact or not
        :return:
        """
        self.kappa = kappa
        self.L2_reg_dual = L2_reg_dual
        self.L2_reg_loss = L2_reg_loss
        self.max_opt_itr = max_opt_itr
        self.tf_optimizer = tf_optimizer
        self.model = model
        self.policy = policy
        self.env = env
        self.dtype = get_default_tf_dtype()
        self.epsilon_small = 1e-24
        self.min_eta_inv = 1e-12
        self.projection_type = projection_type
        self.model_L2_reg_loss = 0
        self.policy_L2_reg_loss = L2_reg_loss
        self.write_every = 1
        self.training_set_size = training_set_size
        self.exact = exact
        self.fit_iterations = fit_iterations
        self.refit_iterations = (
            refit_iterations if not restart_fitting else fit_iterations
        )
        self.restart_fitting = restart_fitting
        self.refit = refit
        self.refit_every_iterations = refit_every_iterations
        self.sess = None
        self.summary_writer = None
        self.global_step = 0
        self.iteration = 0

        # ----------------------------------------
        # placeholders
        # ----------------------------------------
        self.observations_ph = None
        self.actions_one_hot_ph = None
        self.kappa_ph = None  # Constraint on the KL divergence: \kappa
        self.actions_ph = None
        self.rewards_ph = None
        self.returns_ph = None
        self.timesteps_ph = None
        self.next_states_ph = None
        self.feat_diff_ph = None
        self.param_eta = None  # Value of \eta
        self.param_eta_inv_ph = None  # inverse of eta_ 1/\eta
        self.policy_tf = None  # \pi(a | s)
        self.model_tf = None  # p_\omega(s'|s,a)
        self.model_logli = None  # log(p_\omega(s' | s,a))
        self.model_policy_loss = None  #
        self.dual = None  # \min_{\eta\in[0, +\infty)} g(\eta) =
        # \eta \log \ev_{S,A,S' \sim d} \left[ \exp\left(\frac{1}{\eta} r(S,A,S') + \kappa \right) \right]
        self.dual_grad = None  # Gradient of the dual
        self.primal = None  # \exp\left(\frac{1}{\eta} r(s,a,s')\right)
        self.model_grad_loss = None  # Gradient of the loss of the model
        self.policy_grad_loss = None  # Gradient of the loss of the poliicy
        self.model_policy_grad_loss = None  # Gradient of the loss of the model policy
        self.model_loss = None  # Loss of the model
        self.policy_loss = None  # Policy loss
        self.eta = None  # Dual parameter
        self.state_kernel = (
            None
        )  # p\{\theta, \omega} (s' | s) = p_\omega (s'|s,a) \pi_\theta (a | s)

        # ----------------------------------------
        # Optimizers
        # ----------------------------------------
        self.model_policy_tf_optimizer = None
        self.model_tf_optimizer = None
        self.dual_optimizer = None

        # Summary
        self.summarize = None

    def initialize(self, session, summary_writer, omega=5):

        self.sess = session

        self.summary_writer = summary_writer
        self.global_step = 0

        # Tf vars
        self.observations_ph = tf.placeholder(
            dtype=self.dtype, name="obs", shape=(None, self.env.observation_space_size)
        )

        # one hot tensor
        self.actions_one_hot_ph = tf.placeholder(
            dtype=self.dtype, name="action_one_hot", shape=(None, self.env.n_actions)
        )

        self.kappa_ph = tf.placeholder(dtype=self.dtype, name="kappa", shape=())

        # -1, 0, +1 tensor
        # or -1 +1 tensor
        # actual action taken or
        # all actions possible
        # e.g. [-1, 1; -1, 1 ...]
        self.actions_ph = tf.placeholder(
            dtype=self.dtype, name="action", shape=(None, self.env.n_actions)
        )

        self.rewards_ph = tf.placeholder(
            dtype=self.dtype, name="rewards", shape=(None, 1)
        )

        self.returns_ph = tf.placeholder(
            dtype=self.dtype, name="returns", shape=(None,)
        )

        self.timesteps_ph = tf.placeholder(
            dtype=self.dtype, name="timestep", shape=(None,)
        )

        # next state centered on the previous one
        self.next_states_ph = tf.placeholder(
            dtype=self.dtype,
            name="next_states",
            shape=(None, self.env.observation_space_size),
        )

        # Feature difference variable representing the difference in feature
        # value of the next observation and the current observation \phi(s') -
        # \phi(s).
        self.feat_diff_ph = tf.placeholder(
            dtype=self.dtype,
            name="feat_diff",
            shape=(None, self.env.observation_space_size * 2),
        )

        # Get Policy
        policy_tf, _ = self.policy(self.observations_ph)
        model_log_prob_tf, model_prob_tf = self.model(
            self.observations_ph,
            self.actions_ph,
            self.next_states_ph,
            initial_omega=omega,
            training_set_size=self.training_set_size,
            actions_one_hot=self.actions_one_hot_ph,
            sess=session,
            summary_writer=summary_writer,
        )
        self.policy_tf = policy_tf
        self.param_eta_inv_ph = tf.get_variable(name="eta", shape=(), dtype=self.dtype)
        eta = 1 / self.param_eta_inv_ph

        # Model loglikelihood
        model_logli = model_log_prob_tf

        # Policy and model loss loss (KL divergence, to be minimized)
        state_kernel_before_sum = tf.multiply(model_prob_tf, policy_tf)
        state_kernel = tf.reduce_sum(state_kernel_before_sum, axis=1, keepdims=True)

        # algorithm information
        weights = tf.exp(self.rewards_ph / eta - tf.reduce_max(self.rewards_ph / eta))
        weights_norm = weights / tf.reduce_mean(weights)
        max_weights = tf.reduce_max(weights)
        min_weights = tf.reduce_min(weights)
        mean_weights = tf.reduce_mean(weights)

        # For regularization add L2 reg term
        model_policy_loss = -tf.reduce_sum(
            weights * tf.log(state_kernel + self.epsilon_small)
        )

        # add l2 regularization
        # Loss function using L2 Regularization
        regularizers = [tf.reduce_sum(tf.square(x)) for x in self.policy.trainable_vars]
        total_loss = tf.add_n(regularizers)
        model_policy_loss += self.L2_reg_loss * total_loss

        model_policy_grad_loss = tf.gradients(
            model_policy_loss, self.policy.trainable_vars + self.model.trainable_vars
        )
        # bound model param
        var_to_bounds = self.model.get_variables_to_bound()

        # Optimizer
        self.model_policy_tf_optimizer = self.tf_optimizer(
            model_policy_loss,
            var_list=self.model.trainable_vars + self.policy.trainable_vars,
            var_to_bounds=var_to_bounds,
            options={"maxiter": 100, "maxfun": 100},
        )

        # support other type of projection
        # Model kl dvergence
        model_loss = -tf.reduce_mean(weights * model_logli)
        model_regularizers = [
            tf.reduce_sum(tf.square(x)) for x in self.model.trainable_vars
        ]
        model_reg_loss = tf.add_n(model_regularizers)
        model_loss += self.model_L2_reg_loss * model_reg_loss

        model_grad_loss = tf.gradients(model_loss, self.model.trainable_vars)

        self.model_tf_optimizer = self.tf_optimizer(
            model_loss, var_list=self.model.trainable_vars, var_to_bounds=var_to_bounds
        )

        # log of the policy dist
        logli = tf.log(
            tf.reduce_sum(
                tf.multiply(policy_tf, self.actions_one_hot_ph), axis=1, keepdims=True
            )
        )

        # Policy loss (KL divergence, to be minimized)
        policy_loss = -tf.reduce_mean(weights * logli)
        policy_regularizers = [
            tf.reduce_sum(tf.square(x)) for x in self.policy.trainable_vars
        ]
        policy_reg_loss = tf.add_n(policy_regularizers)
        policy_loss += self.policy_L2_reg_loss * policy_reg_loss

        policy_grad_loss = tf.gradients(policy_loss, self.policy.trainable_vars)
        self.policy_tf_optimizer = self.tf_optimizer(
            policy_loss, var_list=self.policy.trainable_vars
        )

        # Dual-related symbolics
        # Symbolic dual
        dual = (
            eta * self.kappa_ph
            + eta * tf.log(tf.reduce_mean(weights))
            + eta * tf.reduce_max(self.rewards_ph / eta)
        )
        # Add L2 regularization.
        dual += self.L2_reg_dual * (
            tf.square(self.param_eta_inv_ph) + tf.square(1 / self.param_eta_inv_ph)
        )

        dual_grad = tf.gradients(dual, [self.param_eta_inv_ph])
        # Set parameter boundaries: \eta>0, v unrestricted.
        dual_bound = {self.param_eta_inv_ph: (self.min_eta_inv, np.infty)}

        var_list = [self.param_eta_inv_ph]
        self.dual_optimizer = self.tf_optimizer(
            dual, var_list=var_list, var_to_bounds=dual_bound, options={"maxiter": 100}
        )

        primal = tf.reduce_mean(weights_norm * self.rewards_ph)

        self.policy_tf = policy_tf
        self.model_logli = model_logli
        self.model_policy_loss = model_policy_loss
        self.dual = dual
        self.dual_grad = dual_grad
        self.primal = primal
        self.model_grad_loss = model_grad_loss
        self.policy_grad_loss = policy_grad_loss
        self.model_policy_grad_loss = model_policy_grad_loss
        self.model_loss = model_loss
        self.policy_loss = policy_loss
        self.eta = eta
        self.state_kernel = state_kernel

        # plot purpose
        mean_ret = tf.reduce_mean(self.returns_ph)
        mean_ts = tf.reduce_mean(self.timesteps_ph)
        ret_sum = tf.summary.scalar("Return", mean_ret)
        ts_sum = tf.summary.scalar("Timesteps", mean_ts)
        model_vars_sum = self.model.get_variable_summaries()
        max_w_sum = tf.summary.scalar("MaxWeights", max_weights)
        min_weight_sum = tf.summary.scalar("MinWeights", min_weights)
        mean_weight_sum = tf.summary.scalar("MeanWeights", mean_weights)
        dual_sum = tf.summary.scalar("Dual", dual)
        primal_sum = tf.summary.scalar("Primal", primal)
        eta_sum = tf.summary.scalar("Eta", self.eta)
        self.summary_writer.add_graph(self.sess.graph)
        self.summarize = tf.summary.merge(
            [
                ret_sum,
                ts_sum,
                max_w_sum,
                min_weight_sum,
                mean_weight_sum,
                dual_sum,
                primal_sum,
                eta_sum,
            ]
            + model_vars_sum
        )
        self.setParamEtaInv = U.SetFromFlat([self.param_eta_inv_ph], dtype=self.dtype)
        self.model_tf = model_prob_tf
        self.iteration = 0

    def _features(self, path):
        o = np.array(path)
        return np.hstack((o, o ** 2))  # , pos*vel, al, al**2, al**3, np.ones((l, 1))))

    def train(self, samples_data: Dict, normalize_rewards: bool = False):
        """

        :param samples_data: contains: rewards
                                       reward_list
                                       actions
                                       timesteps
                                       actions_one_hot
                                       wins
                                       paths
        :param normalize_rewards: boolean, whether to normalize rewards
        :return: The new value of omega
        """
        # Init vars
        rewards = samples_data["rewards"]
        reward_list = samples_data["reward_list"]
        timesteps = samples_data["timesteps"]
        actions_one_hot = samples_data["actions_one_hot"]
        feat_diff = []
        next_states = []
        states = []
        for (i, path) in enumerate(samples_data["paths"]):
            feats = self._features(path)
            obs = np.array(path)
            # all but the first
            if not self.exact:
                # centered
                next_states.append(obs[1:, :] - obs[:-1, :])
            else:
                next_states.append(obs[1:, :])
            # all but the last
            states.append(obs[:-1, :])
            feat_diff.append(feats[1:] - feats[:-1])
        feat_diff = np.vstack(feat_diff)
        states = np.vstack(states)
        next_states = np.vstack(next_states)

        if self.projection_type == "joint":
            actions = np.zeros((states.shape[0], 1))
            if self.env.n_actions == 2:
                actions = np.hstack((actions - 1, actions + 1))
            else:
                actions = np.hstack(
                    (actions + 1, actions + 1, actions + 1, actions + 1)
                )
        else:
            actions = actions_one_hot * [-1, 1]

        if normalize_rewards:
            rewards = (rewards - np.mean(rewards)) / (np.maximum(np.std(rewards), 1e-5))
        assert next_states.shape == states.shape

        inputs_dict = {
            self.rewards_ph: rewards,
            self.actions_one_hot_ph: actions_one_hot,
            self.observations_ph: states,
            self.next_states_ph: next_states,
            self.feat_diff_ph: feat_diff,
            self.actions_ph: actions,
            self.returns_ph: reward_list,
            self.timesteps_ph: timesteps,
            self.kappa_ph: self.kappa,
        }

        inputs_dict.update(self.model.get_feed_dict())

        #################
        # Optimize dual #
        #################
        self.optimize_dual(inputs_dict)

        print(f"Parameters found: {self.param_eta}")

        # save variables before projection
        omega_before = np.array(self.sess.run(self.model.get_omega()))
        th_before = np.array(self.sess.run(self.policy.get_theta()))

        ###################
        # Optimize policy and model #
        ###################
        self.project(inputs_dict)

        # save variable after projection
        omega_after = np.array(
            self.sess.run(self.model.get_omega(), feed_dict=inputs_dict)
        )
        th_after = np.array(self.sess.run(self.policy.get_theta()))

        # log variable
        if self.iteration % self.write_every == 0:
            self.log(
                inputs_dict,
                omega_before,
                th_before,
                omega_after,
                th_after,
                samples_data,
            )

        self.iteration += 1
        self.global_step += 1

        # add samples for refit
        # add to the new training set after subsampling
        # to_add = 1000
        # ind = np.arange(0, np.shape(states)[0])
        # selected_ind = np.random.choice(ind, size=to_add, replace=False)
        # inputs = states[selected_ind, :]
        # ac = np.sum(actions_one_hot[selected_ind,:] * samples_data['omega'], axis=1, keepdims=True)
        # X = np.hstack((inputs, ac))
        # targets = next_states[selected_ind, :]
        # if self.iteration >= 2:
        #     X_old = np.load(self.model.folder+"on_policyX.npy")
        #     targets_old = np.load(self.model.folder+"on_policyY.npy")
        #     X = np.vstack((X_old, X))
        #     targets = np.vstack((targets_old, targets))
        # np.save(self.model.folder + "on_policyX.npy", X)
        # np.save(self.model.folder+"on_policyY.npy", targets)

        if self.iteration % self.refit_every_iterations == 0 and self.refit:
            self.model.fit(
                action_ph=self.actions_ph,
                states_ph=self.observations_ph,
                next_states_ph=self.next_states_ph,
                load_weights=False,
                add_onpolicy=True,
                training_step=1000,
            )

        return omega_after

    def pi(self, state, log=False):
        probs = self.sess.run(self.policy_tf, feed_dict={self.observations_ph: state})[
            0
        ]
        a = np.random.choice(int(self.env.action_space_size), p=probs)
        return a

    def store_data(self, X, Y, normalize_data=False):
        self.model.store_data(X, Y, normalize_data)
        pass

    def fit(self):
        self.model.fit(
            action_ph=self.actions_ph,
            states_ph=self.observations_ph,
            next_states_ph=self.next_states_ph,
            training_step=self.fit_iterations,
            restart_fitting=False,
        )

    def get_policy_params(self):
        return self.policy.trainable_vars

    def optimize_dual(self, inputs_dict):
        # Here we need to optimize dual through BFGS in order to obtain \eta
        # value. Initialize dual function g(\theta, v). \eta > 0
        # Init dual param values
        param_eta_inv = 1.0

        # Initial BFGS parameter values.
        self.setParamEtaInv([param_eta_inv])

        # Optimize through BFGS
        with timed("optimizing dual"):

            self.dual_before, self.dual_grad_before = self.sess.run(
                [self.dual, self.dual_grad], feed_dict=inputs_dict
            )

            self.dual_optimizer.minimize(session=self.sess, feed_dict=inputs_dict)

            self.dual_after, self.grad_after = self.sess.run(
                [self.dual, self.dual_grad], feed_dict=inputs_dict
            )

        # Optimal values have been obtained
        self.param_eta = self.sess.run(self.eta)
        return self.dual_after

    def project(self, inputs_dict):
        # joint or disjoint projection
        if self.projection_type == Projection.STATE_KERNEL:

            # reassign the correct omega
            with timed("projection"):
                self.model_policy_tf_optimizer.minimize(
                    session=self.sess, feed_dict=inputs_dict
                )
        elif self.projection_type == Projection.DISJOINT:
            # Disjoint Projection
            self.model_tf_optimizer.minimize(session=self.sess, feed_dict=inputs_dict)
            self.policy_tf_optimizer.minimize(session=self.sess, feed_dict=inputs_dict)
        elif self.projection_type == Projection.D_PROJECTION:
            raise ValueError("Projection Type not supported")
        else:
            raise ValueError("Error in the definition of projection, see the Projections")

    def log(
        self,
        inputs_dict,
        omega_before,
        theta_before,
        omega_after,
        theta_after,
        samples_data,
    ):
        primal, summary_str = self.sess.run(
            [self.primal, self.summarize], feed_dict=inputs_dict
        )
        delta_variables = theta_after - theta_before
        norm_delta_var = np.linalg.norm(delta_variables)
        delta_omega = omega_after - omega_before
        norm_delta_omega = np.linalg.norm(delta_omega)
        self.summary_writer.add_summary(summary_str, self.global_step)
        # record all
        logger.record_tabular("ITERATIONS", self.iteration)
        logger.record_tabular("Theta", theta_after)
        logger.record_tabular("ThetaBefore", theta_before)
        logger.record_tabular("Primal", primal)
        logger.record_tabular("OmegaBefore", omega_before)
        logger.record_tabular("Omega", omega_after)
        logger.record_tabular("NormDeltaOmega", norm_delta_omega)
        logger.record_tabular("Eta", self.param_eta)
        logger.record_tabular("NormDeltaVar", norm_delta_var)
        logger.record_tabular("DeltaOmega", delta_omega)
        logger.record_tabular("Epsilon", self.kappa)
        logger.record_tabular("ReturnsMean", np.mean(samples_data["reward_list"]))
        logger.record_tabular("ReturnsStd", np.std(samples_data["reward_list"]))
        logger.record_tabular("RewardMean", np.mean(samples_data["reward_list"]))
        logger.record_tabular("RewardStd", np.std(samples_data["rewards"]))
        logger.record_tabular("TimestepsMean", np.mean(samples_data["timesteps"]))
        logger.record_tabular("TimestepsStd", np.std(samples_data["timesteps"]))
        logger.record_tabular("DualReg", self.L2_reg_dual)
        logger.record_tabular("LossReg", self.L2_reg_loss)
        logger.record_tabular("Wins", samples_data["wins"])
        logger.record_tabular("Traj", samples_data["traj"])
        logger.record_tabular("ConfortViolation", samples_data["confort_violation"])
        logger.dump_tabular()
