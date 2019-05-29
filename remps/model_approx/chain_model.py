import numpy as np
import tensorflow as tf

from remps.model_approx.model_approximator import ModelApproximator
from remps.utils.utils import get_default_tf_dtype


class ChainModel(ModelApproximator):
    def set_omega(self):
        pass

    def __init__(self, name="chain"):
        """
        Parameters
        - input_space: dimension of state vector
        """
        self.sess = None
        self.log_prob = None
        self.default_dtype = get_default_tf_dtype()
        self.name = name
        self.omega_value = 0
        self.width = 1
        self.offset = 0
        self.k = 1
        self.param = 0.5

    def __call__(
        self,
        states,
        actions,
        next_states,
        initial_omega=0.8,
        actions_one_hot=None,
        sess=None,
        summary_writer=None,
        **kwargs
    ):

        with tf.variable_scope(self.name):

            self.width = 1
            self.offset = 0
            self.k = 0.07
            self.omega = tf.get_variable(
                dtype=self.default_dtype,
                name="omega",
                shape=(1, 1),
                initializer=tf.initializers.constant(
                    self.from_sigm_to_omega(initial_omega)
                ),
            )

            omega = self.width * (1 / (1 + tf.exp(-self.k * self.omega))) + self.offset

            # state 0, action 0, w, 1-w
            # state 0, action 1, 1-w, w
            # state 1 action 0, w, 1-w
            # state 1 action 1, 1-w, w

            # do for action 0
            first_states_prob = tf.tile(omega, (tf.shape(states)[0], 1))
            second_states_prob = tf.tile(1 - omega, (tf.shape(states)[0], 1))

            next_states_prob_a0 = tf.concat(
                [first_states_prob, second_states_prob], axis=1
            )
            next_states_prob_a0 = tf.reduce_sum(
                tf.multiply(next_states_prob_a0, next_states), axis=1, keepdims=True
            )

            # similar for action 1
            first_states_prob_s0 = tf.tile(1 - omega, (tf.shape(states)[0], 1))
            first_states_prob_s1 = tf.tile(
                1 - omega * self.param, (tf.shape(states)[0], 1)
            )
            first_states_prob = tf.reduce_sum(
                tf.multiply(
                    tf.concat([first_states_prob_s0, first_states_prob_s1], axis=1),
                    states,
                ),
                axis=1,
                keepdims=True,
            )

            second_states_prob_s0 = tf.tile(omega, (tf.shape(states)[0], 1))
            second_states_prob_s1 = tf.tile(
                omega * self.param, (tf.shape(states)[0], 1)
            )
            second_states_prob = tf.reduce_sum(
                tf.multiply(
                    tf.concat([second_states_prob_s0, second_states_prob_s1], axis=1),
                    states,
                ),
                axis=1,
                keepdims=True,
            )

            next_states_prob_a1 = tf.concat(
                [first_states_prob, second_states_prob], axis=1
            )
            next_states_prob_a1 = tf.reduce_sum(
                tf.multiply(next_states_prob_a1, next_states), axis=1, keepdims=True
            )

            self.prob = tf.concat([next_states_prob_a0, next_states_prob_a1], axis=1)

            self.log_prob = tf.log(self.prob)

        return self.log_prob, self.prob

    def store_data(self, X, Y):
        """
        Store training data inside training set
        """
        pass

    def initialize(self, sess):
        self.sess = sess

    def test(self, x, next_states):

        pass

    def get_probability(self):
        return self.log_prob

    def sample_transition(self, x, theta):
        pass

    # fit the gaussian process using XData and YData provided in store data
    def fit(self, **kwargs):
        pass

    # return the feed dict for the optimizer
    def get_feed_dict(self):
        return {}

    def getKernels(self):
        pass

    @property
    def trainable_vars(self):
        return [self.omega]

    def get_variables_to_bound(self):
        return {}

    def from_sigm_to_omega(self, param):
        return -np.log(self.width / (param - self.offset) - 1) / self.k

    def get_omega(self):
        return self.width * (1 / (1 + tf.exp(-self.k * self.omega))) + self.offset

    def get_variable_summaries(self):
        return [tf.summary.scalar("Omega", tf.norm(self.get_omega()))]

    def set_params(self, theta):
        self.theta_value = theta
