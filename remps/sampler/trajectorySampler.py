import os
import os.path
import time
from copy import copy
from multiprocessing import Event, Process, Queue
from multiprocessing.pool import Pool

import numpy as np
import tensorflow as tf

import baselines.common.tf_util as U
from baselines import logger
from baselines.common import set_global_seeds
from remps.utils.utils import get_default_tf_dtype


class SamplingWorker(Process):
    def __init__(self, policy, env, n_traj, inputQ, outputQ, n_actions, obs_size):
        # Invoke parent constructor BEFORE doing anything!!
        Process.__init__(self)
        self.dtype = get_default_tf_dtype()
        self.state_tf = tf.placeholder(self.dtype, (None, obs_size), name="states")
        self.policy = policy
        self.n_traj = n_traj
        self.inputQ = inputQ
        self.outputQ = outputQ
        self.env = copy(env)
        self.n_actions = n_actions

    def run(self):
        """Override Process.run()"""

        # self.env.seed(os.getpid())
        # set_global_seeds(os.getpid())
        self.policy.name = str(os.getpid()) + "policy"
        self.policy_tf, _ = self.policy(self.state_tf)

        # Start TF session
        sess = tf.Session()
        with sess.as_default():
            pi = make_pi(self.policy_tf, sess, self.state_tf, self.n_actions)
            set_parameters = U.SetFromFlat(self.policy.trainable_vars, dtype=self.dtype)
            # Build the sampling logic fn
            sampling_fn = make_sampling_fn(pi, self.env, self.n_traj, self.n_actions)

            # Start sampling-worker loop.
            done = False
            while not done:
                # self.event.wait()  # Wait for a new message
                # self.event.clear()  # Upon message receipt, mark as read
                message, policy_ws, theta = self.inputQ.get()  # Pop message
                if message == "sample":
                    self.env.setParams(theta)
                    # Set weights
                    set_parameters(policy_ws)
                    # Do sampling
                    stats = sampling_fn()
                    self.outputQ.put((os.getpid(), stats))

                elif message == "exit":
                    print("[Worker {}] Exiting...".format(os.getpid()))
                    self.env.close()
                    done = True
                    break
        sess.close()


def make_pi(policy, sess, state_tf, action_size):
    def pi(state):
        probs = sess.run(policy, feed_dict={state_tf: state[np.newaxis, :]})[0]
        a = np.random.choice(int(action_size), p=probs)
        return a

    return pi


def make_sampling_fn(pi, env, n_traj, n_actions):
    # Define the closure
    def sampling_fn():

        states = list()
        next_states = list()
        rewards = list()
        actions_one_hot = list()
        actions = list()
        timesteps = list()
        mask = None

        # statistics
        wins = 0
        reward_list = list()
        paths = list()
        paths_full = {}
        paths_full["states"] = list()
        paths_full["actions"] = list()
        paths_full["next_states"] = list()
        paths_full["rewards"] = list()
        paths_full["actions_one_hot"] = list()
        paths_full["next_states_centred"] = list()
        small_vel = 0
        n = 0
        t = 0
        confort_violation = 0
        print("Sampling....")
        for _ in range(n_traj):
            paths.append(list())
            paths_full["states"].append(list())
            paths_full["actions"].append(list())
            paths_full["next_states"].append(list())
            paths_full["next_states_centred"].append(list())
            paths_full["rewards"].append(list())
            paths_full["actions_one_hot"].append(list())
            rewards_i = list()
            states_i = list()
            next_states_i = list()
            mask_i = list()
            actions_i_one_hot = list()
            actions_i = list()
            done = False

            # gamma_cum is gamma^t
            gamma_cum = 1
            gamma = 1
            cum_reward = 0
            reward = 0
            timesteps_i = 0

            # Sampling logic
            state = env.reset()
            paths[n].append(state)
            paths_full["states"][n].append(state)
            while not done:

                # Select action a_t according to current policy
                a_t = pi(state)

                newState, reward, done, info = env.step(a_t)

                # add to the buffer to remember
                # rewards_i.append(reward*gamma_cum)
                rewards.append(reward * gamma_cum)
                paths_full["rewards"][n].append(reward * gamma_cum)
                paths[n].append(newState)
                if not done:
                    paths_full["states"][n].append(newState)

                # works with two actions
                # actions_i.append(a_t-1)
                if n_actions == 2:
                    actions.append(a_t * 2 - 1)
                    paths_full["actions"][n].append(a_t * 2 - 1)
                else:
                    actions.append(a_t)
                    paths_full["actions"][n].append(a_t)

                # create a one hot vector with the taken action and add to the action matrix
                action_blank = np.zeros(n_actions)
                action_blank[a_t] = 1
                # actions_i_one_hot.append(action_blank)
                actions_one_hot.append(action_blank)
                paths_full["actions_one_hot"][n].append(action_blank)

                # calculation of the reward
                cum_reward += reward * gamma_cum
                gamma_cum = gamma_cum * gamma

                # states_i.append(np.append(np.append(state,action),theta))
                states_i.append(state)
                next_states_i.append(np.array(newState - state))
                paths_full["next_states"][n].append(np.array(newState))
                paths_full["next_states_centred"][n].append(np.array(newState - state))
                state = newState

                timesteps_i += 1
                t += 1
                confort_violation += info.get("confort_violation", 0)

                if info.get("goal_reached", False):
                    # print("Goal reached, state: {}, rew: {}".format(newState, reward))
                    wins += 1
                if info.get("small_vel", False):
                    small_vel += 1

            n += 1

            if n % 10 == 0:
                # print("Done: ", n/nb_episodes*100)
                pass

            states.append(states_i)
            next_states.append(next_states_i)
            # rewards.append(rewards_i)
            timesteps.append(timesteps_i)
            reward_list.append(cum_reward)
            # actions_one_hot.append(actions_i_one_hot)
            # actions.append(actions_i)

        stats = {
            "states": states,
            "next_states": next_states,
            "rewards": rewards,
            "timesteps": timesteps,
            "reward_list": reward_list,
            "actions_one_hot": actions_one_hot,
            "actions": actions,
            "wins": wins,
            "paths": paths,
            "small_vel": small_vel,
            "traj": n,
            "confort_violation": confort_violation,
            "paths_full": paths_full,
        }
        return stats

    return sampling_fn
