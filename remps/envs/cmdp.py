from abc import ABC, abstractmethod


class CMDP(ABC):

    # call this method to set MDP parameters
    @abstractmethod
    def setParams(self, *arg):
        pass
