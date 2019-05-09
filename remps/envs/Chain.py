import gym
from gym import spaces
from gym.utils import seeding
from remps.envs.cmdp import CMDP
import numpy as np


class NChainEnv(gym.Env, CMDP):
    """n-Chain environment
    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward
    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.
    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.
    The observed state is the current state in the chain (0 to n-1).
    This environment is described in section 6.1 of:
    A Bayesian Framework for Reinforcement Learning by Malcolm Strens (2000)
    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """

    def __init__(self, n=2, slip=0.2, small=2, large1=10, large2=8, max_steps=500):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large1 = large1  # payout at end of chain for 'forwards' action
        self.large2 = large2
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.observation_space_size = 2
        self.action_space_size = 2
        self.n_actions = 2
        self.seed()
        self.max_steps = max_steps
        self.steps = 0
        self.param = 0.5

    def setParams(self, omega):
        if np.isscalar(omega):
            self.slip = omega
        else:
            self.slip = omega[0][0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        original_action = action
        self.steps = self.steps + 1
        if action:
            if self.np_random.rand() < self.slip * self.param:
                action = not action  # agent slipped, reverse action taken
        else:
            if self.np_random.rand() < self.slip:
                action = not action  # agent slipped, reverse action taken
        if action:  # 'backwards': go back to the beginning, get small reward
            reward = self.small
            self.state = 0
        elif self.state < self.n - 1:  # 'forwards': go up along the chain
            reward = 0
            self.state += 1
        else:  # 'forwards': stay at the end of the chain, collect large reward
            # here action is 0
            if not original_action:
                # original action was 0
                reward = self.large1
            else:
                # original action was 1, backward
                reward = self.large2
        done = False
        if self.steps >= self.max_steps:
            done = True
        # gamma??
        s = np.zeros(2)
        s[self.state] = 1
        return s, reward, done, {}

    def reset(self):
        self.state = 0
        self.steps = 0
        s = np.zeros(2)
        s[self.state] = 1
        return s
