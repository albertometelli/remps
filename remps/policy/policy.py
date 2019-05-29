from abc import abstractmethod

import tensorflow as tf


class Policy:
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
