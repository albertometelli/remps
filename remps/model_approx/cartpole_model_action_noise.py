import math

import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import \
    MultivariateNormalFullCovariance as mvn

from remps.model_approx.model_approximator import ModelApproximator
from remps.utils.utils import get_default_tf_dtype


class CartPoleModel(ModelApproximator):
    def __init__(self, name="cartpole"):
        """
        Parameters
        - input_space: dimension of state vector
        """
        self.sess = None
        self.log_prob = None
        self.default_dtype = get_default_tf_dtype()
        # must be initialized
        self.name = name
        self.x_range = 4.8
        self.theta_range = 180
        # the noise (3sigma) should be inside 10% of the range
        self.x_var = 1e-6  # (self.x_range/(3*1000))**2
        self.theta_var = 1e-6  # (self.theta_range/(3*1000))**2
        self.x_dot_var = 1e-6  # self.x_var/1e-6
        self.theta_dot_var = 1e-6  # self.theta_var/1e-6
        self.action_noise_var = 1e-2
        self.min_omega = 0.1
        self.max_omega = 30

    def __call__(self, states, actions, next_states, initial_omega=0.2, **kwargs):
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masspole + masscart
        length = 0.5  # actually half the pole's length
        polemass_length = masspole * length
        tau = 0.02  # seconds between state updates

        with tf.variable_scope(self.name):
            self.omega = tf.get_variable(
                dtype=self.default_dtype,
                name="omega",
                shape=(),
                initializer=tf.initializers.constant(initial_omega),
            )

            # x, x_dot, theta, theta_dot = state
            x = tf.expand_dims(states[:, 0], 1)
            x_dot = tf.expand_dims(states[:, 1], 1)
            theta = tf.expand_dims(states[:, 2], 1)
            theta_dot = tf.expand_dims(states[:, 3], 1)

            probs = []
            log_probs = []
            # calculate dynamics for each action in actions
            for i in range(actions.get_shape()[1]):
                forces = self.omega * tf.reshape(actions[:, i], (-1, 1))

                costheta = tf.cos(theta)
                sintheta = tf.sin(theta)
                temp = (
                    forces
                    + polemass_length
                    * tf.multiply(tf.multiply(theta_dot, theta_dot), sintheta)
                ) / total_mass
                thetaacc = (gravity * sintheta - tf.multiply(costheta, temp)) / (
                    length
                    * (
                        4.0 / 3.0
                        - masspole * tf.multiply(costheta, costheta) / total_mass
                    )
                )
                xacc = (
                    temp
                    - polemass_length * tf.multiply(thetaacc, costheta) / total_mass
                )

                # propagate action noise through system dynamics
                b = -costheta / (
                    total_mass
                    * (
                        length
                        * (
                            4.0 / 3.0
                            - masspole * tf.multiply(costheta, costheta) / total_mass
                        )
                    )
                )
                beta = tau * b
                d = polemass_length * costheta / total_mass
                alfa = tau * (1 / total_mass - d * b)
                tau_alfa = tau * alfa
                tau_beta = tau * beta

                x_dot_next = x_dot + tau * xacc
                theta_dot_next = theta_dot + tau * thetaacc
                x_next = x + tau * x_dot_next
                theta_next = theta + tau * theta_dot_next

                # build a multivariate normal distribution with covariance matrix
                mu = [x_next, x_dot_next, theta_next, theta_dot_next]
                loc = tf.concat(mu, axis=1)

                batch_size = tf.shape(x)[0]
                # covariance
                ind_noise = [
                    [self.x_var, 0, 0, 0],
                    [0, self.x_dot_var, 0, 0],
                    [0, 0, self.theta_var, 0],
                    [0, 0, 0, self.theta_dot_var],
                ]

                ind_noise = tf.constant(ind_noise, dtype=self.default_dtype)
                ind_noise = tf.expand_dims(ind_noise, axis=0)
                ind_noise = tf.tile(ind_noise, [batch_size, 1, 1])
                # we need to build:
                # [[0, 0, 0, 0],
                #   [0, alfa^2, 0, alfa*beta]
                #   [0, 0, 0, 0]
                #   [0, alfa*beta, 0, beta^2]
                # ]
                # Notice that we need batch_size matrices like this

                covariance_term_first_row = tf.expand_dims(
                    tf.concat(
                        [
                            tau_alfa ** 2,
                            tau_alfa * alfa,
                            tau_alfa * tau_beta,
                            tau_alfa * beta,
                        ],
                        axis=1,
                    ),
                    axis=1,
                )
                covariance_term_second_row = tf.expand_dims(
                    tf.concat(
                        [alfa * tau_alfa, alfa ** 2, alfa * tau_beta, alfa * beta],
                        axis=1,
                    ),
                    axis=1,
                )
                covariance_term_third_row = tf.expand_dims(
                    tf.concat(
                        [
                            tau_beta * tau_alfa,
                            tau_beta * alfa,
                            tau_beta ** 2,
                            tau_beta * beta,
                        ],
                        axis=1,
                    ),
                    axis=1,
                )
                covariance_term_fourth_row = tf.expand_dims(
                    tf.concat(
                        [beta * tau_alfa, alfa * beta, beta * tau_beta, beta ** 2],
                        axis=1,
                    ),
                    axis=1,
                )
                cov_matrix = tf.concat(
                    [
                        covariance_term_first_row,
                        covariance_term_second_row,
                        covariance_term_third_row,
                        covariance_term_fourth_row,
                    ],
                    axis=1,
                )
                action_noise_var = self.action_noise_var
                self.cov_matrix_full = cov_matrix * action_noise_var + ind_noise

                distr = mvn(
                    loc=loc, covariance_matrix=self.cov_matrix_full, validate_args=True
                )

                next_states_prob = tf.expand_dims(distr.prob(next_states), axis=1)
                probs.append(next_states_prob)
                next_states_log_prob = tf.expand_dims(
                    distr.log_prob(next_states), axis=1
                )
                log_probs.append(next_states_log_prob)
            log_prob = tf.concat(log_probs, axis=1)
            prob = tf.concat(probs, axis=1)

        return log_prob, prob

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
    def getFeedDict(self, x, theta, next_states):
        feed_dict = {}
        next_states = next_states.astype(np.float64)
        d_dict = {
            self.x: x,
            self.theta: self.theta_value,
            self.next_states: next_states,
            self.is_test: False,
        }
        return d_dict

    def getKernels(self):
        pass

    def get_feed_dict(self):
        return {}

    def get_variables_to_bound(self):
        return {self.omega: (self.min_omega, self.max_omega)}

    def get_variable_summaries(self):
        return [tf.summary.scalar("Omega", self.omega)]

    @property
    def trainable_vars(self):
        return [self.omega]

    def get_omega(self):
        return self.omega

    def set_omega(self, theta):
        pass

    def set_params(self, theta):
        self.theta_value = theta
