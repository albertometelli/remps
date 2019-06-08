import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import baselines.common.tf_util as U
from baselines import logger
from remps.utils.utils import (flat_and_pad, get_default_tf_dtype,
                               get_tf_optimizer)


class GPOMDP:
    """
    Model and policy Optimizer implementation using GPOMDP gradient estimate
    """

    def __init__(
        self,
        model,
        policy,
        clip_gradient=False,
        env=None,
        n_trajectories=100,
        exact=False,
    ):

        self.model = model
        self.policy = policy
        self.clip_gradient = clip_gradient
        self.dtype = get_default_tf_dtype()
        self.env = env
        self.global_step = 0
        self.iteration = 0
        self.n_trajectories = n_trajectories
        self.exact = exact

    def initialize(self, sess, summary_writer, omega=5):

        self.summary_writer = summary_writer
        self.sess = sess

        # placeholders
        self.mask = U.get_placeholder("mask", self.dtype, (None, 1))

        # Tf vars
        self.observations_ph = U.get_placeholder(
            dtype=self.dtype, name="obs", shape=(None, self.env.observation_space_size)
        )

        # one hot tensor
        self.actions_one_hot_ph = U.get_placeholder(
            name="action_one_hot",
            dtype=self.dtype,
            shape=(None, self.env.action_space_size),
        )

        # -1, 0, +1 tensor
        # or -1 +1 tensor
        # actual action taken or
        # all actions possible
        # e.g. [-1, 1; -1, 1 ...]
        self.actions_ph = U.get_placeholder(
            name="action", dtype=self.dtype, shape=(None, self.env.n_actions)
        )

        self.rewards_ph = U.get_placeholder(
            dtype=self.dtype, name="rewards", shape=(None, 1)
        )

        self.returns_ph = U.get_placeholder(
            name="returns", dtype=self.dtype, shape=(None,)
        )

        self.timesteps_ph = U.get_placeholder(
            name="timestep", dtype=self.dtype, shape=(None,)
        )

        # next state centered on the previous one
        self.next_states_ph = U.get_placeholder(
            name="next_states",
            dtype=self.dtype,
            shape=(None, self.env.observation_space_size),
        )

        self.optimizer = get_tf_optimizer("adam")
        theta = np.random.rand()
        policy_tf, log_prob_policy = self.policy(self.observations_ph, theta)
        model_log_prob_tf, model_prob_tf = self.model(
            self.observations_ph,
            self.actions_ph,
            self.next_states_ph,
            initial_omega=omega,
            actions_one_hot=self.actions_one_hot_ph,
            sess=sess,
            summary_writer=summary_writer,
        )

        policy_prob_taken_ac = tf.reduce_sum(
            policy_tf * self.actions_one_hot_ph, axis=1, keepdims=True
        )
        model_prob_taken_ac = tf.reduce_sum(
            model_prob_tf * self.actions_one_hot_ph, axis=1, keepdims=True
        )
        log_prob = tf.log(model_prob_taken_ac * policy_prob_taken_ac + 1e-20)

        # split using trajectory size
        splitted_probs = tf.concat(
            tf.split(tf.transpose(log_prob), self.n_trajectories, axis=1), axis=0
        )
        splitted_mask = tf.concat(
            tf.split(tf.transpose(self.mask), self.n_trajectories, axis=1), axis=0
        )
        splitted_reward = tf.concat(
            tf.split(tf.transpose(self.rewards_ph), self.n_trajectories, axis=1), axis=0
        )

        # this is the cumulative sum from 0 to t for each timestep t along each trajectory
        cum_sum_probs = tf.cumsum(splitted_probs, axis=1)

        # apply the mask
        cum_sum_probs = tf.multiply(cum_sum_probs, splitted_mask)

        # product between p and discounted reward
        p_times_rew = tf.multiply(cum_sum_probs, splitted_reward)

        # sum over the timesteps
        sum_H = tf.reduce_sum(p_times_rew, axis=1, name="Final_sum_over_timesteps")

        # mean over episodes
        mean_N = tf.reduce_mean(sum_H, axis=0)

        # compute and apply gradients
        self.grad = tf.gradients(
            mean_N, self.model.trainable_vars + self.policy.trainable_vars
        )

        # Summary things
        self.sum_reward = tf.reduce_sum(splitted_reward, axis=1)
        self.mean_reward = tf.reduce_mean(self.sum_reward)
        self.mean_timesteps = tf.reduce_mean(self.timesteps_ph)

        # plot purpose
        mean_ret = tf.reduce_mean(self.returns_ph)
        mean_ts = tf.reduce_mean(self.timesteps_ph)
        ret_sum = tf.summary.scalar("Return", mean_ret)
        ts_sum = tf.summary.scalar("Timesteps", mean_ts)
        om_sum = tf.summary.scalar("Omega", tf.norm(self.model.get_omega()))
        # th_sum = tf.summary.scalar("Theta",tf.norm(self.policy.getTheta()))
        self.summary_writer.add_graph(sess.graph)
        self.summarize = tf.summary.merge([ret_sum, ts_sum, om_sum])  # th_sum])

        # minimize op
        # change sign since we want to maximize
        self.minimize_op = self.optimizer.minimize(
            -mean_N, var_list=self.model.trainable_vars + self.policy.trainable_vars
        )
        self.policy_tf = policy_tf
        self.model_tf = model_prob_tf
        self.log_prob = log_prob

    def train(self, samples_data, normalize_rewards=True):

        rewards = samples_data["rewards"]
        reward_list = samples_data["reward_list"]
        actions = samples_data["actions"]
        timesteps = samples_data["timesteps"]
        actions_one_hot = samples_data["actions_one_hot"]
        wins = samples_data.get("wins", 0)
        observations = samples_data["observations"]
        obs_flat_and_padded = flat_and_pad(samples_data["paths_full"]["states"])
        if not self.exact:
            next_states_flat_and_padded = flat_and_pad(
                samples_data["paths_full"]["next_states_centred"]
            )
        else:
            next_states_flat_and_padded = flat_and_pad(
                samples_data["paths_full"]["next_states"]
            )

        # preprocess rewards
        if normalize_rewards:
            mean_rew = np.mean(rewards)
            std_rew = np.maximum(np.std(rewards), 1e-5)
            for i in range(len(samples_data["paths_full"]["rewards"])):
                samples_data["paths_full"]["rewards"][i] = (
                    samples_data["paths_full"]["rewards"][i] - mean_rew
                ) / std_rew

        rew_flat_and_padded = flat_and_pad(samples_data["paths_full"]["rewards"])
        mask = rew_flat_and_padded != 0
        actions_one_hot_flat_and_padded = flat_and_pad(
            samples_data["paths_full"]["actions_one_hot"]
        )
        actions = np.zeros((obs_flat_and_padded.shape[0], 1))
        actions = np.hstack((actions - 1, actions + 1))

        omega_before = self.sess.run(self.model.get_omega())
        variables_before = U.GetFlat(self.policy.trainable_vars)()

        inputs_dict = {
            self.rewards_ph: rew_flat_and_padded,
            self.actions_one_hot_ph: actions_one_hot_flat_and_padded,
            self.observations_ph: obs_flat_and_padded,
            self.next_states_ph: next_states_flat_and_padded,
            self.actions_ph: actions,
            self.returns_ph: reward_list,
            self.timesteps_ph: timesteps,
            self.mask: mask,
        }

        inputs_dict.update(self.model.get_feed_dict())

        _, summary_str, ac_prob, model_prob, log_prob = self.sess.run(
            [
                self.minimize_op,
                self.summarize,
                self.policy_tf,
                self.model_tf,
                self.log_prob,
            ],
            feed_dict=inputs_dict,
        )
        self.global_step += 1
        self.summary_writer.add_summary(summary_str, self.global_step)

        omega = self.sess.run(self.model.get_omega())
        variables_after = U.GetFlat(self.policy.trainable_vars)()

        delta_variables = variables_after - variables_before
        norm_delta_var = np.linalg.norm(delta_variables)
        delta_omega = omega - omega_before
        norm_delta_omega = np.linalg.norm(delta_omega)
        theta = self.sess.run(self.policy.get_theta())
        # record all
        logger.record_tabular("ITERATIONS", self.iteration)
        logger.record_tabular("Theta", theta[0, 0])
        logger.record_tabular("OmegaBefore", omega_before[0, 0])
        logger.record_tabular("Omega", omega[0, 0])
        logger.record_tabular("NormDeltaOmega", norm_delta_omega)
        logger.record_tabular("NormDeltaVar", norm_delta_var)
        logger.record_tabular("DeltaOmega", delta_omega)
        logger.record_tabular("ReturnsMean", np.mean(reward_list))
        logger.record_tabular("ReturnsStd", np.std(reward_list))
        logger.record_tabular("RewardMean", np.mean(rewards))
        logger.record_tabular("RewardStd", np.std(rewards))
        logger.record_tabular("TimestepsMean", np.mean(timesteps))
        logger.record_tabular("TimestepsStd", np.std(timesteps))
        logger.record_tabular("Wins", wins)
        logger.record_tabular("Traj", samples_data["traj"])
        logger.record_tabular("ConfortViolation", samples_data["confort_violation"])
        logger.dump_tabular()
        self.iteration += 1
        return omega

    def pi(self, state, log=False):
        probs = self.sess.run(self.policy_tf, feed_dict={self.observations_ph: state})[
            0
        ]
        a = np.random.choice(int(self.env.action_space_size), p=probs)
        return a

    def storeData(self, X, Y, normalize_data=False):
        self.model.store_data(X, Y, normalize_data)
        pass

    def fit(self):
        self.model.fit(
            action_ph=self.actions_ph,
            states_ph=self.observations_ph,
            next_states_ph=self.next_states_ph,
        )

    def get_policy_params(self):
        return self.policy.trainable_vars

    def get_model_params(self):
        return self.model.trainable_vars
