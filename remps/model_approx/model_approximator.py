from abc import ABC, abstractmethod

import numpy as np


class ModelApproximator(ABC):

    @abstractmethod
    @property
    def trainable_vars(self):
        pass

    @abstractmethod
    def store_data(self, X: np.array, Y: np.array, **kwargs):
        """
        Store transitions
        :param X: starting state, action and params
        :param Y: Delta state
        :return:
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_probability(self):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fit the Model using X and Y provided in store data
        :param args: additional arguments
        :return:
        """
        pass

    @abstractmethod
    def get_omega(self):
        pass

    @abstractmethod
    def set_omega(self):
        pass
