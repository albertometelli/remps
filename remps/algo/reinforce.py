import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import baselines.common.tf_util as U
from baselines import logger
from remps.utils.utils import flat_and_pad, get_default_tf_dtype, get_tf_optimizer


class REINFORCE:
    """
    Model and policy Optimizer implementation using REINFORCE gradient estimate
    """

    def __init__(
        self, model, policy, clip_gradient=False, env=None, n_trajectories=100
    ):

        self.model = model
        self.policy = policy
        self.clip_gradient = clip_gradient
        self.dtype = get_default_tf_dtype()
        self.env = env
        self.global_step = 0
        self.iteration = 0
        self.n_trajectories = n_trajectories

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

        policy_tf, log_prob_policy = self.policy(self.observations_ph)
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
        list_log_prob = tf.split(tf.transpose(log_prob), self.n_trajectories, axis=1)
        splitted_probs = tf.concat(list_log_prob, axis=0)
        splitted_mask = tf.concat(
            tf.split(tf.transpose(self.mask), self.n_trajectories, axis=1), axis=0
        )
        splitted_reward = tf.concat(
            tf.split(tf.transpose(self.rewards_ph), self.n_trajectories, axis=1), axis=0
        )

        gradient_list = [
            tf.gradients(x, tf.trainable_variables()) for x in list_log_prob
        ]
        # flatten each element in gradient list
        gradient_list_flatten = [list() for _ in range(len(gradient_list))]
        for i, episode_grad in enumerate(gradient_list):
            # reshape each gradient and concatenate
            gradient_list_flatten[i] = tf.concat(
                [tf.reshape(x, (1, -1)) for x in gradient_list[i]], axis=1
            )

        # size split is used to reconstruct all
        size_splits = tf.convert_to_tensor(
            [tf.reshape(x, (1, -1)).get_shape()[1].value for x in gradient_list[0]]
        )

        print([tf.reshape(x, (1, -1)).get_shape()[1].value for x in gradient_list[0]])
        print("Size splits (flattened variables: ", size_splits)
        # now gradient_list_flatten contains a list of tensor 1xK, concatenate
        # gradients is a tensor: N x number of variables, each row is the gradient of an episode
        # wrt all variables
        gradients_matrix = tf.concat(gradient_list_flatten, axis=0)

        gradients_squared = tf.square(gradients_matrix)

        # N rows 1 column
        sum_reward = tf.reduce_sum(splitted_reward, axis=1, keepdims=True)

        baseline_num = tf.reduce_mean(
            tf.multiply(gradients_squared, sum_reward), axis=0
        )

        # 1 row x #variables
        baseline = tf.divide(baseline_num, tf.reduce_mean(gradients_squared, axis=0))

        print("Baseline shape: (should be k) ", baseline.get_shape())

        grad_with_baseline = tf.reduce_mean(
            gradients_matrix * (sum_reward - baseline), axis=0
        )

        print("Grad with baseline shape: ", grad_with_baseline)

        grad_with_baseline_list = tf.split(grad_with_baseline, size_splits)

        print(
            "First element of reconstructed gradients, should be a scalar: ",
            grad_with_baseline[0].get_shape(),
        )

        print("Grad with baseline shape: (should be K)", grad_with_baseline)

        grads_and_vars = self.optimizer.compute_gradients(
            splitted_probs, tf.trainable_variables()
        )

        grads_and_vars_processed = []

        for i, couple in enumerate(grads_and_vars):
            grad, var = couple
            # IMPORTANT: Change the sign of the gradient since the optimizer minimizes
            newgrad = tf.reshape(-grad_with_baseline_list[i], grad.get_shape())
            grads_and_vars_processed.append((newgrad, var))

        self.train_op = self.optimizer.apply_gradients(
            grads_and_vars_processed, self.global_step
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
        om_sum = tf.summary.scalar("Omega", tf.norm(self.model.getOmega()))
        # th_sum = tf.summary.scalar("Theta",tf.norm(self.policy.getTheta()))
        for grad, var in grads_and_vars:
            tf.summary.histogram(var.name, var)
            if grad is not None:
                tf.summary.histogram(var.name + "/gradients", grad)
        self.summary_writer.add_graph(sess.graph)
        self.summarize = tf.summary.merge_all()

        # minimize op
        # change sign since we want to maximize
        self.policy_tf = policy_tf
        self.model_tf = model_prob_tf
        self.log_prob = log_prob

    def train(self, samples_data, normalize_rewards=False):

        rewards = samples_data["rewards"]
        reward_list = samples_data["reward_list"]
        actions = samples_data["actions"]
        timesteps = samples_data["timesteps"]
        actions_one_hot = samples_data["actions_one_hot"]
        wins = samples_data.get("wins", 0)
        observations = samples_data["observations"]
        obs_flat_and_padded = flat_and_pad(samples_data["paths_full"]["states"])
        next_states_flat_and_padded = flat_and_pad(
            samples_data["paths_full"]["next_states_centred"]
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

        omega_before = self.sess.run(self.model.getOmega())
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
                self.train_op,
                self.summarize,
                self.policy_tf,
                self.model_tf,
                self.log_prob,
            ],
            feed_dict=inputs_dict,
        )
        self.global_step += 1
        self.summary_writer.add_summary(summary_str, self.global_step)

        omega = self.sess.run(self.model.getOmega())
        variables_after = U.GetFlat(self.policy.trainable_vars)()

        delta_variables = variables_after - variables_before
        norm_delta_var = np.linalg.norm(delta_variables)
        delta_omega = omega - omega_before
        norm_delta_omega = np.linalg.norm(delta_omega)
        # theta = self.sess.run(self.policy.getTheta())
        # record all
        logger.record_tabular("ITERATIONS", self.iteration)
        # logger.record_tabular("Theta", theta)
        logger.record_tabular("OmegaBefore", omega_before)
        logger.record_tabular("Omega", omega)
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
        self.model.storeData(X, Y, normalize_data)
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
