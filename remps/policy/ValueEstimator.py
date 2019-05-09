import tensorflow as tf


class ValueEstimator:
    def __init__(
        self,
        state_space,
        action_space,
        hidden_layer_size,
        sess,
        scope="value_estimator",
    ):

        with tf.variable_scope(scope):
            # placeholders
            self.state = tf.placeholder(
                tf.float32, (None, state_space), name="policy-state"
            )
