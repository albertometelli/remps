from abc import ABC, abstractmethod


class ModelApprox(ABC):
    @property
    def trainable_vars(self):
        pass

    # call this method to store transitions:
    # X: starting state, action and params
    # Y: Delta state
    @abstractmethod
    def storeData(self, X, Y):
        pass

    @abstractmethod
    def getProb(self):
        pass

    # fit the gaussian process using XData and YData provided in store data
    @abstractmethod
    def fit(self, *args):
        pass

    @abstractmethod
    def getOmega(self):
        pass

    @abstractmethod
    def setOmega(self):
        pass
