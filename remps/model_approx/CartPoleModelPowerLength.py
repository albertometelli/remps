import tensorflow as tf
import numpy as np
from remps.utils.logger import log
from sklearn.utils.validation import check_X_y, check_array
from remps.model_approx.modelApprox import ModelApprox
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance as mvn
import math


class CartPoleModel(ModelApprox):
    def __init__(self, name="cartpole"):
        """
        Parameters
        - input_space: dimension of state vector
        """
        self.sess = None
        self.log_prob = None
        self.default_dtype = tf.float32
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
        self.min_length = 0.1
        self.max_length = 1

    def __call__(
        self, states, actions, next_states, initial_omega=0.2, initial_length=0.5
    ):
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masspole + masscart
        # length = 0.5 # actually half the pole's length
        # polemass_length = (masspole * length)
        tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        theta_threshold_radians = 12 * 2 * math.pi / 360
        x_threshold = 2.4

        with tf.variable_scope(self.name):
            self.omega = tf.get_variable(
                dtype=self.default_dtype,
                name="omega",
                shape=(1, 1),
                initializer=tf.initializers.constant(initial_omega),
            )

            self.length = tf.get_variable(
                dtype=self.default_dtype,
                name="length",
                shape=(),
                initializer=tf.initializers.constant(initial_length),
            )

            length = self.length  # actually half the pole's length
            polemass_length = masspole * length

            # x, x_dot, theta, theta_dot = state
            x = tf.expand_dims(states[:, 0], 1)
            x_dot = tf.expand_dims(states[:, 1], 1)
            theta = tf.expand_dims(states[:, 2], 1)
            theta_dot = tf.expand_dims(states[:, 3], 1)

            probs = []
            log_probs = []
            # calculate dynamics for each action in actions
            for i in range(actions.get_shape()[1]):
                forces = self.omega * tf.expand_dims(actions[:, i], axis=1)

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

    def storeData(self, X, Y, normalize_data=None):
        """
        Store training data inside training set
        """
        pass

    def initialize(self, sess):
        self.sess = sess

    def test(self, x, next_states):

        pass

    def getProb(self):
        return self.log_prob

    def sample_transition(self, x, theta):
        pass

    # fit the model
    def fit(self, **kwargs):
        pass

    # return the feed dict for the optimizer
    def get_feed_dict(self):
        return {}

    def getKernels(self):
        pass

    @property
    def trainable_vars(self):
        return [self.omega, self.length]

    def get_variable_to_bound(self):
        return {
            self.omega: (self.min_omega, self.max_omega),
            self.length: (self.min_length, self.max_length),
        }

    def getOmega(self):
        return self.omega

    def get_variable_summaries(self):
        return [
            tf.summary.scalar("Omega", self.omega),
            tf.summary.scalar("Length", self.length),
        ]

    def setOmega(self, theta):
        pass

    def set_params(self, theta):
        self.theta_value = theta
