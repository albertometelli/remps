import numpy as np
import tensorflow as tf

from remps.policy.policy import Policy
from remps.utils.utils import get_default_tf_dtype


class Gaussian(Policy):
    """
    Used for torcs
    MultiLayerPerceptron Discrete policy.
    Parametrized by the input space, the action space and the hidden layer size.
    Basic policy network with only one hidden layer with sigmoid activation function
    """

    def __init__(self, state_space, action_space, hidden_layer_size, name="policy"):
        """
        Builds a policy network and returns a node for the gradient and a node for action selection
        Simple network: from state space to action space
        Start from a random policy, all weights equal to 0
        @param state_space: dimension of state space
        @param actions_space: dimension of action space
        @param trajectory_size: number of trajectories collected for estimating the gradient
        @param checkpoint_file: name of checkpoint file in which to save variables
        @param restore: True if need to restore variables
        """
        # net params
        super().__init__(name)
        self.hidden_layer_size = hidden_layer_size
        self.state_space = state_space
        self.action_space = action_space
        self.sess = None
        self.default_dtype = get_default_tf_dtype()

    def __call__(self, state, taken_actions):

        with tf.variable_scope(self.name):
            # Net
            self.eps = tf.constant(1e-24, dtype=self.default_dtype)
            if self.hidden_layer_size > 0:
                biases = tf.get_variable(
                    "b",
                    [self.hidden_layer_size],
                    initializer=tf.random_normal_initializer(
                        0, 0.001, dtype=self.default_dtype
                    ),
                    dtype=self.default_dtype,
                )
                W = tf.get_variable(
                    "W",
                    [self.state_space, self.hidden_layer_size],
                    initializer=tf.random_normal_initializer(
                        0, 0.001, dtype=self.default_dtype
                    ),
                    dtype=self.default_dtype,
                )

                h = tf.matmul(state, W)
                h = tf.tanh(h + biases)

            else:
                h = state

            steer = tf.layers.dense(
                inputs=h, units=1, activation=tf.tanh, use_bias=True
            )
            acc = tf.layers.dense(
                inputs=h, units=1, activation=tf.sigmoid, use_bias=True
            )
            brake = tf.layers.dense(
                inputs=h, units=1, activation=tf.sigmoid, use_bias=True
            )
            v_steer = tf.exp(
                tf.get_variable(
                    "v_steer",
                    1,
                    initializer=tf.random_normal_initializer(
                        0, 0.1, dtype=self.default_dtype
                    ),
                    dtype=self.default_dtype,
                )
            )
            v_acc = tf.exp(
                tf.get_variable(
                    "v_acc",
                    1,
                    initializer=tf.random_normal_initializer(
                        0, 0.1, dtype=self.default_dtype
                    ),
                    dtype=self.default_dtype,
                )
            )
            v_brake = tf.exp(
                tf.get_variable(
                    "v_brake",
                    1,
                    initializer=tf.random_normal_initializer(
                        0, 0.1, dtype=self.default_dtype
                    ),
                    dtype=self.default_dtype,
                )
            )

            means = tf.concat([steer, acc, brake])
            stds = tf.concat([v_steer, v_acc, v_brake])
            self.dist = tf.distributions.Normal(means, stds)
            self._pi = self.dist.sample()
            self._pi_prob = self.dist.prob(taken_actions)
            self._log_pi = self.dist.log_prob(taken_actions)
        return self._pi_prob, self._log_pi

    def pi(self, s, log=True):
        """
        Selects an action according to the net probability
        @param s: state vector
        @param log: if True log probabilities
        """
        ac = self.sess.run(self._pi, feed_dict={self.state: s})[0]
        return ac

    def get_policy_network(self):
        return self._pi

    def initialize(self, sess):
        self.sess = sess
        init = tf.initialize_variables(self.trainable_vars)
        self.sess.run(init)
