import numpy as np
import tensorflow as tf

from remps.policy.Model import Model
from remps.utils.utils import get_default_tf_dtype


class MLPDiscrete(Model):
    """
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
        self.hidden_layer_size = hidden_layer_size
        self.state_space = state_space
        self.action_space = action_space
        self.name = name
        self.sess = None
        self.default_dtype = get_default_tf_dtype()

    def __call__(self, state):

        with tf.variable_scope(self.name):
            # Net
            self.eps = tf.constant(1e-24, dtype=self.default_dtype)
            if self.hidden_layer_size > 0:
                hidden_layer_size = self.hidden_layer_size
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
                hidden_layer_size = self.state_space
                h = state

            biases2 = tf.get_variable(
                "b2",
                [self.action_space],
                # initializer=tf.random_normal_initializer(0,0.001, dtype=self.default_dtype),
                initializer=tf.constant_initializer(0),
                dtype=self.default_dtype,
            )
            W2 = tf.get_variable(
                "W2",
                [hidden_layer_size, self.action_space],
                # initializer=tf.random_normal_initializer(0,0.001, dtype=self.default_dtype),
                initializer=tf.constant_initializer(0),
                dtype=self.default_dtype,
            )

            h2 = tf.matmul(h, W2)
            self.h2 = h2 + biases2

            # For taking actions
            self._pi = tf.nn.softmax(self.h2)
            print("PI SHAPE: ", self._pi.get_shape())
            self._log_pi = tf.log(self._pi + self.eps)
        return self._pi, self._log_pi

    def pi(self, s, log=True):
        """
        Selects an action according to the net probability
        @param s: state vector
        @param log: if True log probabilities
        """
        probs = self.sess.run(self._pi, feed_dict={self.state: s})[0]

        if log:
            print(probs)
        if np.isnan(probs[0]):
            print("NAN")

        # draw a sample according to probabilities
        a = np.random.choice(int(self.action_space), p=probs)

        return a

    def get_policy_network(self):
        return self._pi

    def initialize(self, sess):
        self.sess = sess
        init = tf.initialize_variables(self.trainable_vars)
        self.sess.run(init)
