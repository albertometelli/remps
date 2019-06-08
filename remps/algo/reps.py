"""
Relative entropy policy model search
Reference: https://pdfs.semanticscholar.org/ff47/526838ce85d77a50197a0c5f6ee5095156aa.pdf
Idea: use REPS to find the distribution p(s,a,s') containing both policy and transition model.
Then matches the distributions minimizing the KL between the p and the induced distribution from
\pi and \p_\omega
Follows the rllab implementation
"""

from copy import copy

import numpy as np
import scipy.optimize
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

import baselines
import baselines.common.tf_util as U
from baselines import logger


class REPMS:
    """
    Relative Entropy Policy Search (REPS)

    References
    ----------
    [1] J. Peters, K. Mulling, and Y. Altun, "Relative Entropy Policy Search," Artif. Intell., pp. 1607-1612, 2008.

    """

    def __init__(
        self,
        epsilon=1e-3,  # 0.001,
        L2_reg_dual=0.0,  # 1e-5,
        L2_reg_loss=0.0,
        L2_reg_projection=0,
        max_opt_itr=1000,
        optimizer=scipy.optimize.fmin_l_bfgs_b,
        tf_optimizer=ScipyOptimizerInterface,
        model=None,
        policy=None,
        env=None,
        projection_type="joint",  # joint or disjoint, joint: state kernel projection
        **kwargs
    ):
        """
        :param epsilon: Max KL divergence between new policy and old policy.
        :param L2_reg_dual: Dual regularization
        :param L2_reg_loss: Loss regularization
        :param max_opt_itr: Maximum number of batch optimization iterations.
        :param optimizer: Module path to the optimizer. It must support the same interface as
        scipy.optimize.fmin_l_bfgs_b.
        :return:
        """
        super(REPMS, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.L2_reg_dual = L2_reg_dual
        self.L2_reg_loss = L2_reg_loss
        self.max_opt_itr = max_opt_itr
        self.optimizer = optimizer
        self.tf_optimizer = tf_optimizer
        self.opt_info = None
        self.model = model
        self.policy = policy
        self.env = env
        self.dtype = tf.float64
        self.epsilon_small = 1e-20
        self.projection_type = projection_type
        self.model_L2_reg_loss = 0
        self.policy_L2_reg_loss = L2_reg_loss
        self.L2_reg_projection = L2_reg_projection

    def initialize(self, session, summary_writer, theta=5):

        self.sess = session
        # Init dual param values
        self.param_eta = 1.0
        # Adjust for linear feature vector.
        self.param_v = np.random.rand(self.env.observation_space_size * 2 + 1)

        self.summary_writer = summary_writer
        self.global_step = 0

        # Tf vars
        self.observations_ph = tf.placeholder(
            dtype=self.dtype, name="obs", shape=(None, self.env.observation_space_size)
        )

        # one hot tensor
        self.actions_one_hot_ph = tf.placeholder(
            dtype=self.dtype,
            name="action_one_hot",
            shape=(None, self.env.action_space_size),
        )

        # -1, 0, +1 tensor
        # or -1 +1 tensor
        # actual action taken or
        # all actions possible
        # e.g. [-1, 1; -1, 1 ...]
        self.actions_ph = tf.placeholder(
            dtype=self.dtype,
            name="action",
            shape=(None, self.env.n_actions if self.projection_type == "joint" else 1),
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
            shape=(None, self.env.observation_space_size * 2 + 1),
        )
        theta = np.random.rand()
        policy_tf, _ = self.policy(self.observations_ph, theta)
        other_policy = copy(self.policy)
        other_policy.name = "OtherPolicy"
        other_policy_tf, _ = other_policy(self.observations_ph, theta)
        self.policy_tf = policy_tf
        self.param_v_ph = tf.placeholder(
            name="param_v",
            shape=(self.env.observation_space_size * 2 + 1, 1),
            dtype=self.dtype,
        )
        self.param_eta_ph = tf.placeholder(name="eta", shape=(), dtype=self.dtype)

        # Symbolic sample Bellman error
        delta_v = self.rewards_ph + tf.matmul(self.feat_diff_ph, self.param_v_ph)
        print("Delta v: ", delta_v.get_shape())

        print("Policy: ", policy_tf.get_shape())

        # Policy and model loss loss (KL divergence, to be minimized)
        state_kernel_before_sum = tf.multiply(model_prob_tf, policy_tf)
        other_state_kernel_before_sum = tf.multiply(
            other_model_prob_tf, other_policy_tf
        )
        state_kernel = tf.reduce_sum(state_kernel_before_sum, axis=1, keepdims=True)
        other_state_kernel = tf.reduce_sum(
            other_state_kernel_before_sum, axis=1, keepdims=True
        )
        weights = tf.exp(
            delta_v * self.param_eta_ph - tf.reduce_max(delta_v * self.param_eta_ph)
        )
        weights_exp = delta_v * self.param_eta_ph - tf.reduce_max(
            delta_v * self.param_eta_ph
        )
        print("State kernel shape: ", state_kernel.get_shape())

        # For regularization add L2 reg term
        # use sum
        model_policy_loss = -tf.reduce_mean(
            tf.exp(
                (
                    delta_v * self.param_eta_ph
                    - tf.reduce_max(delta_v * self.param_eta_ph)
                )
            )
            * tf.log(state_kernel + self.epsilon_small)
        )

        # add l2 regularization
        # var_loss = # Loss function using L2 Regularization
        regularizers = [tf.reduce_sum(tf.square(x)) for x in self.policy.trainable_vars]
        total_loss = tf.add_n(regularizers)
        print("Reg loss shape:", total_loss.get_shape())
        model_policy_loss += self.L2_reg_loss * (total_loss)
        print("model policy loss", model_policy_loss.get_shape())
        model_policy_loss += self.L2_reg_projection * tf.add_n(
            [
                tf.reduce_sum(tf.square(x - y))
                for x, y in zip(self.policy.trainable_vars, other_policy.trainable_vars)
            ]
        )

        self.model_policy_tf_optimizer = self.tf_optimizer(
            model_policy_loss,
            var_list=self.model.trainable_vars + self.policy.trainable_vars,
        )

        # log of the policy dist
        logli = tf.log(
            tf.reduce_sum(
                tf.multiply(policy_tf, self.actions_one_hot_ph), axis=1, keepdims=True
            )
        )
        print("Policy: ", logli.get_shape())

        # Policy loss (KL divergence, to be minimized)
        policy_loss = -tf.reduce_mean(
            logli
            * tf.exp(
                delta_v * self.param_eta_ph - tf.reduce_max(delta_v * self.param_eta_ph)
            )
        )
        policy_regularizers = [
            tf.reduce_sum(tf.square(x)) for x in self.policy.trainable_vars
        ]
        policy_reg_loss = tf.add_n(policy_regularizers)
        policy_loss += self.policy_L2_reg_loss * (policy_reg_loss)

        print("Policy loss shape: ", policy_loss.get_shape())
        print("Policy vars", self.policy.trainable_vars)
        self.policy_tf_optimizer = self.tf_optimizer(
            policy_loss, var_list=self.policy.trainable_vars
        )

        # Dual-related symbolics
        # Symbolic dual
        # debug purposes
        inside_log = tf.reduce_mean(
            tf.exp(
                delta_v * self.param_eta_ph - tf.reduce_max(delta_v) * self.param_eta_ph
            )
        )
        inside_log_f = U.function(
            inputs=[self.rewards_ph, self.feat_diff_ph]
            + [self.param_eta_ph, self.param_v_ph],
            outputs=inside_log,
        )

        # (1/self.param_eta_ph) * self.epsilon +
        dual = (
            (1 / self.param_eta_ph) * self.epsilon
            + (1 / self.param_eta_ph) * tf.log(inside_log)  # + self.epsilon_small
            + tf.reduce_max(delta_v)
        )
        # Add L2 regularization.
        dual += self.L2_reg_dual * (
            tf.square(self.param_eta_ph) + tf.square(1 / self.param_eta_ph)
        )  # + tf.reduce_sum(tf.square(self.param_v_ph)))

        # Symbolic dual gradient
        dual_grad = U.flatgrad(dual, [self.param_eta_ph, self.param_v_ph])

        # Eval functions.
        f_dual = U.function(
            inputs=[self.rewards_ph, self.feat_diff_ph]
            + [self.param_eta_ph, self.param_v_ph],
            outputs=dual,
        )
        f_dual_grad = U.function(
            inputs=[self.rewards_ph, self.feat_diff_ph]
            + [self.param_eta_ph, self.param_v_ph],
            outputs=dual_grad,
        )

        max_delta_v_eta = tf.reduce_max(delta_v * self.param_eta_ph)
        exp_delta_v_eta_minus_max = tf.exp(
            delta_v * self.param_eta_ph - max_delta_v_eta
        )
        mean_delta_v_eta_minus_max = tf.reduce_mean(exp_delta_v_eta_minus_max)
        exp_delta_v_eta = tf.exp(delta_v * self.param_eta_ph)
        mean_delta_v_eta = tf.reduce_mean(exp_delta_v_eta)

        # KL(p||q) = (1/E[deltav]) * E(delta_v*(eta^-1))
        d_kl_pq = (
            tf.reduce_mean((delta_v * self.param_eta_ph) * exp_delta_v_eta_minus_max)
            / mean_delta_v_eta_minus_max
            - tf.log(mean_delta_v_eta_minus_max)
            - max_delta_v_eta
        )
        d_kl_pq_2 = tf.reduce_mean(
            delta_v * self.param_eta_ph * exp_delta_v_eta
        ) / mean_delta_v_eta - tf.log(mean_delta_v_eta)
        d_kl_p_hat_q = tf.reduce_mean(
            tf.log(state_kernel + self.epsilon_small)
            - tf.log(other_state_kernel + self.epsilon_small)
        )

        self.opt_info = dict(
            f_dual=f_dual,
            f_dual_grad=f_dual_grad,
            model_policy_loss=model_policy_loss,
            policy_loss=policy_loss,
            model_loss=model_loss,
            inside_log=inside_log_f,
            state_kernel=state_kernel,
            delta_v=delta_v,
            d_kl_pq=d_kl_pq,
            d_kl_pq_2=d_kl_pq_2,
            d_kl_p_hat_q=d_kl_p_hat_q
            # model_grad = model_f_loss_grad
        )

        self.policy_tf = policy_tf
        self.model_logli = model_logli
        self.model_policy_loss = model_policy_loss
        self.weights = weights
        self.weights_exp = weights_exp

        # plot purpose
        mean_ret = tf.reduce_mean(self.returns_ph)
        mean_ts = tf.reduce_mean(self.timesteps_ph)
        tf.summary.scalar("Reward", mean_ret)
        tf.summary.scalar("Timesteps", mean_ts)
        tf.summary.scalar("Theta", tf.reduce_sum(self.model.get_theta()))
        tf.summary.scalar("KL", d_kl_pq)
        tf.summary.scalar("KL2", d_kl_pq_2)
        self.summary_writer.add_graph(self.sess.graph)
        self.summarize = tf.summary.merge_all()
        self.setModelParam = U.SetFromFlat(other_model.trainable_vars, dtype=tf.float64)
        self.setPolicyParam = U.SetFromFlat(
            other_policy.trainable_vars, dtype=tf.float64
        )
        self.other_policy_tf = other_policy_tf
        self.other_model_tf = other_model_log_prob_tf
        self.other_model = other_model
        self.other_policy = other_policy

    def _features(self, path):
        o = np.array(path)
        pos = np.expand_dims(o[:, 0], 1)
        vel = np.expand_dims(o[:, 1], 1)
        # l = np.shape(path)[0]
        # al = np.arange(l).reshape(-1, 1) / 1000.0
        return np.hstack(
            (o, o ** 2, pos * vel)
        )  # al, al**2, al**3))#, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    # convert to the usage of my framework
    # samples data contains:
    # - rewards
    # - observations : visited states
    # - paths : list of observations, used to build next states and states
    #           (add to observation all but the last states)
    # - actions: taken actions
    # - actions_one_hot: one hot vector of taken actions
    def train(self, samples_data, normalize_rewards=False):
        # Init vars
        rewards = samples_data["rewards"]
        reward_list = samples_data["reward_list"]
        actions = samples_data["actions"]
        timesteps = samples_data["timesteps"]
        actions_one_hot = samples_data["actions_one_hot"]
        # unused
        observations = samples_data["observations"]
        # agent_infos = samples_data["agent_infos"]
        # Compute sample Bellman error.
        feat_diff = []
        next_states = []
        states = []
        for (i, path) in enumerate(samples_data["paths"]):
            feats = self._features(path)
            obs = np.array(path)
            # all but the first
            # centered
            next_states.append(obs[1:, :])
            # all but the last
            states.append(obs[:-1, :])
            # feats = np.vstack([feats, np.zeros(feats.shape[1])])
            feat_diff.append(feats[1:] - feats[:-1])
        feat_diff = np.vstack(feat_diff)
        states = np.vstack(states)

        if self.projection_type == "joint":
            actions = np.zeros((states.shape[0], 1))
            actions = np.hstack((actions - 1, actions + 1))

        next_states = np.vstack(next_states)
        if normalize_rewards:
            rewards = (rewards - np.min(rewards)) / (np.min(rewards) - np.max(rewards))
        assert next_states.shape == states.shape
        print("actions Shape", actions.shape)

        #################
        # Optimize dual #
        #################

        # Here we need to optimize dual through BFGS in order to obtain \eta
        # value. Initialize dual function g(\theta, v). \eta > 0
        # First eval delta_v
        f_dual = self.opt_info["f_dual"]
        f_dual_grad = self.opt_info["f_dual_grad"]
        inside_log_f = self.opt_info["inside_log"]

        # Set BFGS eval function
        def eval_dual(input):
            param_eta = input[0]
            param_v = np.matrix(input[1:]).transpose()
            if param_eta == 0.0:
                return +np.inf
            # print("Parameters: ", param_eta, param_v)
            # inside_log_val = inside_log_f(*([rewards, feat_diff] + [param_eta, param_v]))
            # print("Inside log", inside_log_val)
            val = f_dual(*([rewards, feat_diff] + [param_eta, param_v]))
            # print("Function value", val)
            return val.astype(np.float64)

        # Set BFGS gradient eval function
        def eval_dual_grad(input):
            param_eta = input[0]
            param_v = np.matrix(input[1:]).transpose()
            grad = f_dual_grad(*([rewards, feat_diff] + [param_eta, param_v]))
            # eta_grad = np.matrix(np.float(grad[0]))
            # v_grad = np.transpose(grad[1])
            # grad = np.hstack([eta_grad, v_grad])
            # print("Gradient", np.expand_dims(grad,axis=0).transpose())
            return np.expand_dims(grad, axis=0).transpose()

        # Initial BFGS parameter values.
        x0 = np.hstack([self.param_eta, self.param_v])

        # Set parameter boundaries: \eta>0, v unrestricted.
        bounds = [(-np.inf, np.inf) for _ in x0]
        bounds[0] = (1e-24, np.inf)  # for numerical reasons

        # Optimize through BFGS
        logger.log("optimizing dual")
        eta_before = x0[0]
        v_before = x0[1:]
        dual_before = eval_dual(x0)
        params_ast, _, d = self.optimizer(
            func=eval_dual,
            x0=x0,
            fprime=eval_dual_grad,
            bounds=bounds,
            maxiter=self.max_opt_itr,
            disp=0,
            factr=10.0,
        )

        print(d)

        dual_after = eval_dual(params_ast)
        grad_after = eval_dual_grad(params_ast)
        print(
            "Dual after optimization: {}, Grad after optimization {}".format(
                dual_after, grad_after
            )
        )

        # Optimal values have been obtained
        param_eta = params_ast[0]
        param_v = params_ast[1:]

        print("Parameters found: {}, {}".format(param_eta, param_v))

        ###################
        # Optimize policy and model #
        ###################
        model_policy_loss = self.opt_info["model_policy_loss"]
        policy_loss = self.opt_info["policy_loss"]
        model_loss = self.opt_info["model_loss"]
        state_kernel = self.opt_info["state_kernel"]
        delta_v = self.opt_info["delta_v"]
        theta_before = self.sess.run(self.model.get_theta())[0]

        logger.log("optimizing policy")
        inputs_dict = {
            self.rewards_ph: rewards,
            self.actions_one_hot_ph: actions_one_hot,
            self.observations_ph: states,
            self.next_states_ph: next_states,
            self.param_eta_ph: param_eta,
            self.param_v_ph: np.transpose(np.expand_dims(param_v, 0)),
            self.feat_diff_ph: feat_diff,
            self.actions_ph: actions,
        }

        # calculate KL between state kernels of p and old p
        d_kl_p_hat_q = self.opt_info["d_kl_p_hat_q"]
        d_kl_p_hat_q_value = self.sess.run(d_kl_p_hat_q, inputs_dict)
        print(
            "-------------------- KL after projection ---------------- \n KL :{}".format(
                d_kl_p_hat_q_value
            )
        )

        # save old variables before variable optimization
        self.setPolicyParam(U.GetFlat(self.policy.trainable_vars)())
        self.setModelParam(U.GetFlat(self.model.trainable_vars)())

        def policy_step_callback(arg):
            print("optimization step Vars:", arg)
            pass

        def policy_loss_callback(**kwarg):
            print("optimization step Loss:", kwarg)
            pass

        if self.projection_type == "joint":
            grad_loss = tf.gradients(
                model_policy_loss,
                self.policy.trainable_vars + self.model.trainable_vars,
            )
            model_policy_loss_grad, model_policy_loss_value, state_kernel_val, variables, delta_v_val, weights, weights_exp = self.sess.run(
                [
                    grad_loss,
                    model_policy_loss,
                    state_kernel,
                    self.policy.trainable_vars + self.model.trainable_vars,
                    delta_v,
                    self.weights,
                    self.weights_exp,
                ],
                feed_dict=inputs_dict,
            )
            variables_before = variables
            print(state_kernel_val[state_kernel_val == 0])
            print(actions[actions == 0])
            print(weights[weights != 0.0])
            print(weights_exp)
            print(
                "Before model policy optimization: \n Loss: {} \n Gradient: {} \n Variables: {} \n State kernel: {} \n Delta v: {} \n Weights :{}".format(
                    model_policy_loss_value,
                    model_policy_loss_grad,
                    variables,
                    state_kernel_val,
                    delta_v_val,
                    weights,
                )
            )

            self.model_policy_tf_optimizer.minimize(
                session=self.sess,
                feed_dict=inputs_dict,
                step_callback=policy_step_callback,
                loss_callback=policy_loss_callback,
            )
            model_policy_loss_grad, model_policy_loss_value, state_kernel_val, variables, delta_v_val = self.sess.run(
                [
                    grad_loss,
                    model_policy_loss,
                    state_kernel,
                    self.policy.trainable_vars + self.model.trainable_vars,
                    delta_v,
                ],
                feed_dict=inputs_dict,
            )
            print(state_kernel_val[state_kernel_val == 0])
            print(actions[actions == 0])
            print(
                "After model policy optimization: \n Loss: {} \n Gradient: {} \n Variables: {} \n State kernel: {}\n".format(
                    model_policy_loss_value,
                    model_policy_loss_grad,
                    variables,
                    state_kernel_val,
                )
            )
        else:
            grad_policy = tf.gradients(policy_loss, self.policy.trainable_vars)
            grad_model = tf.gradients(model_loss, self.model.trainable_vars)
            policy_loss_grad, policy_loss_value, model_loss_grad, model_loss_value, variables, deltav_val = self.sess.run(
                [
                    grad_policy,
                    policy_loss,
                    grad_model,
                    model_loss,
                    self.policy.trainable_vars + self.model.trainable_vars,
                    delta_v,
                ],
                feed_dict=inputs_dict,
            )
            variables_before = variables
            print(
                "Before model policy optimization: \n Policy Loss: {} \n Model Loss: {} \n Policy Gradient: {} \n Model Gradient: {} \n Variables: {} \n Delta v {}".format(
                    policy_loss_value,
                    model_loss_value,
                    policy_loss_grad,
                    model_loss_grad,
                    variables,
                    deltav_val,
                )
            )

            # self.model_tf_optimizer.minimize(session=self.sess,
            #                                         feed_dict=inputs_dict,
            #                                         step_callback=policy_step_callback)
            self.policy_tf_optimizer.minimize(
                session=self.sess,
                feed_dict=inputs_dict,
                step_callback=policy_step_callback,
                loss_callback=policy_loss_callback,
            )

            policy_loss_grad, policy_loss_value, model_loss_grad, model_loss_value, variables, deltav_val = self.sess.run(
                [
                    grad_policy,
                    policy_loss,
                    grad_model,
                    model_loss,
                    self.policy.trainable_vars + self.model.trainable_vars,
                    delta_v,
                ],
                feed_dict=inputs_dict,
            )
            print(
                "After model policy optimization: \n Policy Loss: {} \n Model Loss: {} \n Policy Gradient: {} \n Model Gradient: {} \n Variables: {} \n Delta v {}".format(
                    policy_loss_value,
                    model_loss_value,
                    policy_loss_grad,
                    model_loss_grad,
                    variables,
                    deltav_val,
                )
            )

        logger.log("eta %f -> %f" % (eta_before, param_eta))
        logger.log("V {} -> {}".format(v_before, param_v))

        logger.record_tabular("DualBefore", dual_before)
        logger.record_tabular("DualAfter", dual_after)
        logger.log("Dual %f -> %f " % (dual_before, dual_after))

        theta_after = self.sess.run(self.model.get_theta())[0]
        logger.record_tabular("ThetaAfter", theta_before)
        logger.log("theta %f -> %f" % (theta_before, theta_after))
        delta_variables = [x - y for (x, y) in zip(variables, variables_before)]
        delta_theta = theta_after - theta_before
        print("Delta var: {}, delta theta: {}".format(delta_variables, delta_theta))

        # calculate mean KL between new and old state kernel
        d_kl_pq = self.opt_info["d_kl_pq"]
        d_kl_pq_2 = self.opt_info["d_kl_pq_2"]
        kl = self.sess.run(d_kl_pq, feed_dict=inputs_dict)
        kl_2 = self.sess.run(d_kl_pq_2, feed_dict=inputs_dict)
        logger.log("KL Divergence: %f" % (kl))
        logger.log("KL2 Divergence: %f" % (kl_2))
        # plot
        inputs_dict.update({self.returns_ph: reward_list, self.timesteps_ph: timesteps})
        summary_str = self.sess.run(self.summarize, feed_dict=inputs_dict)
        self.global_step += 1
        self.summary_writer.add_summary(summary_str, self.global_step)
        return theta_after[0]

    def pi(self, state, log=False):
        probs = self.sess.run(self.policy_tf, feed_dict={self.observations_ph: state})[
            0
        ]
        a = np.random.choice(int(self.env.action_space_size), p=probs)
        return a

    def storeData(self, X, Y):
        pass

    def get_policy_params(self):
        return self.policy.trainable_vars
