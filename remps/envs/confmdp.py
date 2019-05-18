from abc import abstractmethod

import numpy as np
import gym


class ConfMDP(gym.Env):

    @abstractmethod
    def set_params(self, *arg):
        """
        call this method to set MDP parameters
        :param arg: parameters to set
        :return: None
        """
        pass

    @abstractmethod
    def get_params_bounds(self) -> np.array:
        """
        For each parameter get the bounds
        Returns a Nx2 matrix in which for
        each parameter we have min and max
        :return: None
        """
        pass

    @abstractmethod
    @property
    def observation_space_size(self) -> int:
        """
        Returns the size of the observation space
        :return:
        """
        pass

    @abstractmethod
    def get_params(self) -> np.array:
        """
        Returns the environment parameters
        :return: the environment parameters
        """
        pass
