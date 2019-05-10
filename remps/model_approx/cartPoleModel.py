import math

import numpy as np
import tensorflow as tf

from remps.model_approx.modelApprox import ModelApprox


class CartPoleModel(ModelApprox):
    def __init__(self, name="cartpole"):
        """
        Parameters
        - input_space: dimension of state vector
        """
        self.sess = None
        self.log_prob = None
        self.default_dtype = tf.float64
        # must be initialized
        self.name = name

    def __call__(self, states, actions, next_states, initial_omega=0.2):
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masspole + masscart
        length = 0.5  # actually half the pole's length
        polemass_length = masspole * length
        tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        theta_threshold_radians = 12 * 2 * math.pi / 360
        x_threshold = 2.4
        self.x_std = tf.constant(1e-3, dtype=tf.float64)
        self.x_dot_std = tf.constant(1e-4, dtype=tf.float64)
        self.theta_std = tf.constant(1e-3, dtype=tf.float64)
        self.theta_dot_std = tf.constant(1e-4, dtype=tf.float64)

        with tf.variable_scope(self.name):
            self.omega = tf.get_variable(
                dtype=self.default_dtype,
                name="omega",
                shape=(1, 1),
                initializer=tf.initializers.constant(initial_omega),
            )
            # limited range (sigmoid)
            omega = (
                self.omega
            )  # self.width*(1/(1+tf.exp(-self.k*self.omega)))+self.offset

            # x, x_dot, theta, theta_dot = state
            x = tf.expand_dims(states[:, 0], 1)
            x_dot = tf.expand_dims(states[:, 1], 1)
            theta = tf.expand_dims(states[:, 2], 1)
            theta_dot = tf.expand_dims(states[:, 3], 1)

            next_x = tf.expand_dims(next_states[:, 0], 1)
            next_x_dot = tf.expand_dims(next_states[:, 1], 1)
            next_theta = tf.expand_dims(next_states[:, 2], 1)
            next_theta_dot = tf.expand_dims(next_states[:, 3], 1)

            # should be n x 2 vector of actions times vel
            forces = self.omega * actions

            costheta = tf.cos(theta)
            sintheta = tf.sin(theta)
            temp = (
                forces
                + polemass_length
                * tf.multiply(tf.multiply(theta_dot, theta_dot), sintheta)
            ) / total_mass
            thetaacc = (gravity * sintheta - tf.multiply(costheta, temp)) / (
                length
                * (4.0 / 3.0 - masspole * tf.multiply(costheta, costheta) / total_mass)
            )
            xacc = temp - polemass_length * tf.multiply(thetaacc, costheta) / total_mass
            x = x + tau * x_dot
            x_dot = x_dot + tau * xacc
            theta = theta + tau * theta_dot
            theta_dot = theta_dot + tau * thetaacc

            pdf_x = tf.distributions.Normal(x, scale=self.x_std)
            pdf_x_dot = tf.distributions.Normal(x_dot, scale=self.x_dot_std)
            pdf_theta = tf.distributions.Normal(theta, scale=self.theta_std)
            pdf_theta_dot = tf.distributions.Normal(theta_dot, scale=self.theta_dot_std)

            prob_x = pdf_x.prob(next_x)
            prob_x_dot = pdf_x_dot.prob(next_x_dot)
            prob_theta = pdf_theta.prob(next_theta)
            prob_theta_dot = pdf_theta_dot.prob(next_theta_dot)

            log_prob_x = pdf_x.log_prob(next_x)
            log_prob_x_dot = pdf_x_dot.log_prob(next_x_dot)
            log_prob_theta = pdf_theta.log_prob(next_theta)
            log_prob_theta_dot = pdf_theta_dot.log_prob(next_theta_dot)

            log_prob = log_prob_x + log_prob_x_dot + log_prob_theta + log_prob_theta_dot
            prob = prob_x * prob_x_dot * prob_theta * prob_theta_dot

        return log_prob, prob

    def storeData(self, X, Y):
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

    # fit the gaussian process using XData and YData provided in store data
    def fit(self, use_scikit_fit=True):
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

    @property
    def trainable_vars(self):
        return [self.omega]

    def getOmega(self):
        return self.omega

    def setOmega(self, theta):
        pass
        # self.theta_value = np.matrix(theta)

    def set_params(self, theta):
        self.theta_value = theta
