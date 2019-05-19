import numpy as np
import tensorflow as tf

from remps.policy.policy import Policy
from remps.utils.utils import get_default_tf_dtype


class OneParameterPolicy(Policy):
    """
    Policy defined by one param: theta, prob of action 0
    """

    def __init__(self, name="policy", init_theta=np.random.rand()):
        """
        Builds a policy network and returns a node for pi and a node for logpi
        """
        # net params
        super(OneParameterPolicy, self).__init__(name)
        self.sess = None
        self.default_dtype = get_default_tf_dtype()
        self.epsilon_small = 1e-20
        self.action_space = 2
        self.init_theta = init_theta

    def __call__(self, state):

        with tf.variable_scope(self.name):
            # Net
            self.width = 1
            self.offset = 0
            self.k = 0.07
            self.theta = tf.get_variable(
                "theta",
                dtype=get_default_tf_dtype(),
                shape=(1, 1),
                initializer=tf.initializers.constant(
                    self.from_sigm_to_theta(self.init_theta)
                ),
            )

            theta = self.width * (1 / (1 + tf.exp(-self.k * self.theta))) + self.offset

            # For taking actions
            self._pi = tf.concat(
                [
                    tf.tile(theta, (tf.shape(state)[0], 1)),
                    tf.tile(1 - theta, (tf.shape(state)[0], 1)),
                ],
                axis=1,
            )

            self._log_pi = tf.log(self._pi + self.epsilon_small)

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

    def from_sigm_to_theta(self, param):
        return -np.log(self.width / (param - self.offset) - 1) / self.k

    def getTheta(self):
        return self.width * (1 / (1 + tf.exp(-self.k * self.theta))) + self.offset

    def get_policy_network(self):
        return self._pi

    def initialize(self, sess):
        self.sess = sess
        init = tf.initialize_variables(self.trainable_vars)
        self.sess.run(init)
