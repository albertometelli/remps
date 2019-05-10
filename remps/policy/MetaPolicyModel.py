import numpy as np
import tensorflow as tf

from remps.policy.Model import Model


class MetaPolicyModel:
    """
    Probability distribution (Gaussian) over policy and model parameters
    Mean and std are treated as variables since they need to be optimized
    """

    def __init__(
        self, sess, name="metaPmetaM", policy_param=6, model_param=1, scale=1e-3
    ):
        self.sess = sess
        self.name = name
        self.scale = scale
        self.policy_param = policy_param
        self.model_param = model_param

    def __call__(self, samples=None):
        with tf.variable_scope(self.name):
            policy_scales = tf.constant(
                self.scale, dtype=tf.float64, shape=(1, self.policy_param)
            )
            model_scales = tf.constant(
                self.scale * 1e2, dtype=tf.float64, shape=(1, self.model_param)
            )
            # parameters are the mean of the distributions
            self.params = tf.get_variable(
                "param",
                dtype=tf.float64,
                shape=(1, self.policy_param + self.model_param),
                initializer=tf.initializers.constant(0),
            )
            self.dist = tf.distributions.Normal(
                self.params, tf.concat([policy_scales, model_scales], axis=1)
            )
            if samples is not None:
                self.prob = tf.reduce_prod(self.dist.prob(samples), axis=1)
                self.log_prob = tf.reduce_sum(self.dist.log_prob(samples), axis=1)
                return self.prob, self.log_prob
            else:
                return None

    def sample(self, n):
        return self.sess.run(self.dist.sample([n]))

    def initPolicyModel(self):
        params = self.sample(1)
        policy_params = params[0, 0, 0 : self.policy_param]
        model_params = params[0, 0, self.policy_param :]
        return params[0, 0], policy_params, model_params[0] + 8

    def trainable_vars(self):
        return [self.params]
