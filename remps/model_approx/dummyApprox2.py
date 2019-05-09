import tensorflow as tf
import numpy as np
from remps.utils.logger import log
from sklearn.utils.validation import check_X_y, check_array
from remps.model_approx.modelApprox import ModelApprox


class MountainCarDummyApprox(ModelApprox):
    """
    Use the mountain car dynamics.
    The model is a gaussian with this mean and some variance
    force = action*power/mass
    velocity += force + math.cos(3*position)*(-0.0025)
    position += velocity + noise
    action must be a tensor with the actual action taken (-1, 0, +1)
    """

    def __init__(self, param_space, name="gp", mass=5000, noise_std=0.01):
        """
        Parameters
        - input_space: dimension of state vector
        """
        self.sess = None
        self.log_prob = None
        self.gp_list = []
        self.default_dtype = tf.float64
        # must be initialized
        self.name = name
        self.theta_value = 0
        self.mass_value = mass
        self.param_space = param_space
        self.width = 20
        self.offset = 0
        self.k = 0.07
        self.noise_std = noise_std

    def __call__(self, states, actions, next_states, omega=None, initial_omega=10):

        with tf.variable_scope(self.name):

            self.mass = tf.constant(self.mass_value, dtype=self.default_dtype)
            self.max_speed = tf.constant(0.1, dtype=self.default_dtype)
            self.pos_std = tf.constant(
                self.noise_std, dtype=self.default_dtype
            )  # 0.0001
            self.vel_std = tf.constant(
                self.noise_std, dtype=self.default_dtype
            )  # 0.00001
            if omega is None:
                omega = tf.get_variable(
                    dtype=self.default_dtype,
                    name="omega",
                    shape=(1, self.param_space),
                    initializer=tf.initializers.constant(initial_omega),
                )
            self.omega = omega
            # limited range (sigmoid)
            omega = (
                self.omega
            )  # self.width*(1/(1+tf.exp(-self.k*self.theta)))+self.offset

            pos = tf.expand_dims(states[:, 0], 1)
            vel = tf.expand_dims(states[:, 1], 1)

            next_pos = tf.expand_dims(next_states[:, 0], 1)
            next_vel = tf.expand_dims(next_states[:, 1], 1)

            log("actions: " + str(actions.get_shape()))
            log("shape" + str(tf.shape(states)[0]))
            # build the action vector
            self.omega_vec = tf.multiply(
                tf.tile(omega, (tf.shape(states)[0], self.param_space)), actions
            )

            force = tf.divide(self.omega_vec, self.mass)

            velDelta = force + tf.cos(3 * pos) * (-0.0025)  # tf.cos(3*pos)*(-0.0025)

            newVel = vel + velDelta

            newVel = tf.clip_by_value(newVel, -self.max_speed, self.max_speed)

            newPos = pos + newVel

            newPos = tf.clip_by_value(newPos, -1.2, 10)  # no need for upper clipping

            # velDelta = tf.clip_by_value(velDelta, -self.max_speed, self.max_speed)

            print("Vel delta shape:", velDelta.get_shape())

            self.pdf_pos = tf.distributions.Normal(newPos, scale=self.pos_std)

            self.pdf_vel = tf.distributions.Normal(newVel, scale=self.vel_std)

            self.prob_pos = self.pdf_pos.prob(next_pos)
            self.prob_vel = self.pdf_vel.prob(next_vel)

            self.log_prob = self.pdf_pos.log_prob(next_pos) + self.pdf_vel.log_prob(
                next_vel
            )
            self.prob = self.pdf_pos.prob(next_pos) * self.pdf_vel.prob(next_vel)

            log("latest prob" + str(self.log_prob.get_shape()))
        return self.log_prob, self.prob

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
