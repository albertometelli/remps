import copy
import os.path
import time
from collections import deque
from datetime import datetime
from multiprocessing import Event, Process, Queue
from multiprocessing.pool import Pool

import baselines.common.tf_util as U
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from baselines.common.tf_util import GetFlat, SetFromFlat
from tensorflow.python import debug as tf_debug

from remps.algo.gpomdp import GPOMDP
from remps.algo.gradient_descent import Adam
from remps.algo.remps_chain import REPMS
from remps.envs.chain import NChainEnv
from remps.policy.discrete import Discrete
from remps.runners.envRunner import runEnv
from remps.sampler.trajectorySampler import SamplingWorker


def trainModelPolicy(
    env, policy, model_approximator, n_trajectories=20, theta=5, **kwargs
):
    n_actions = env.action_space.n

    writer = tf.summary.FileWriter("chain" + str(time.time()))

    # setup agent
    agent = GPOMDP(
        policy=policy, model=model_approximator, env=env, n_trajectories=n_trajectories
    )

    with U.single_threaded_session() as sess:
        # initialization with session
        agent.initialize(sess, writer, theta)

        # to save variables
        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        sess.run(init)

        # make sure all variables are initialized
        sess.run(tf.assert_variables_initialized())

        print("Collecting Data")

        done = False

        reward_mean_to_plot = list()
        reward_std_to_plot = list()

        # set env params
        env.set_params(0.8)

        get_policy_params = U.GetFlat(agent.get_policy_params())

        get_model_params = U.GetFlat(agent.get_model_params())

        thetas = []

        omegas = []

        reward_l = []

        sampling_fn = make_sampling_fn(agent.pi, env, n_trajectories, n_actions)

        for n in range(20):
            # we need to build three vectors that are the concatenations of state, actions and rewards for each trajectory
            states = list()
            next_states = list()
            rewards = list()
            actions_one_hot = list()
            actions = list()
            timesteps = list()
            paths = list()
            mask = None

            # statistics
            wins = 0
            small_vel = 0
            reward_list = list()

            start_time = time.time()

            stats = sampling_fn()

            states = stats["states"]
            paths = stats["paths"]
            next_states = stats["next_states_one_hot"]
            rewards = stats["rewards"]
            actions_one_hot = stats["actions_one_hot"]
            actions = stats["actions"]
            timesteps = stats["timesteps"]
            reward_list = stats["reward_list"]
            feat_diff = stats["feat_diff"]

            # print time
            print("Sampling requires {:.2f} seconds".format(time.time() - start_time))

            start_time = time.time()

            samples_data = {
                "actions": np.matrix(actions).transpose(),
                "actions_one_hot": np.array(actions_one_hot),
                "next_states": next_states,
                "observations": states,
                "paths": paths,
                "rewards": np.transpose(np.expand_dims(np.array(rewards), axis=0)),
                "reward_list": reward_list,
                "timesteps": timesteps,
                "feat_diff": feat_diff,
                "states": states,
            }

            thetas.append(get_policy_params()[0])
            omegas.append(get_model_params()[0])
            reward_l.append(np.mean(reward_list))

            # learning routine
            omega = agent.train(samples_data)

            print("Training requires {:.2f} seconds".format(time.time() - start_time))

            env.set_params(omega)

            # print statistics
            print("Training steps: ", n)
            print("Number of wins: ", wins)
            print("Percentage of wins: ", (wins / n_trajectories) * 100)
            print("Average reward: ", np.mean(reward_list))
            print("Win with small vel: ", small_vel)

        # Close the env
        env.close()
    print(thetas)
    print(omegas)
    # plt.plot(thetas, omegas,'bo-')
    # plt.show()
    # plt.plot(np.arange(0,len(reward_l)), reward_l,'bo-')
    # plt.show()
    return thetas, omegas, reward_l


def make_sampling_fn(pi, env, nb_episodes, n_actions):
    # Define the closure
    def sampling_fn():

        states = list()
        next_states = list()
        rewards = list()
        actions_one_hot = list()
        next_states_one_hot = list()
        actions = list()
        timesteps = list()
        feat_diff = list()
        mask = None

        # statistics
        wins = 0
        reward_list = list()
        paths = list()
        small_vel = 0
        feats = np.eye(2)

        for n in range(nb_episodes):
            paths.append(list())
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
            while not done:
                s = np.zeros((1, 2))
                s[0, state] = 1
                # Select action a_t according to current policy
                a_t = pi(s)

                newState, reward, done, info = env.step(a_t)

                # add to the buffer to remember
                # rewards_i.append(reward*gamma_cum)
                rewards.append(reward * gamma_cum)
                paths[n].append(newState)

                # works with two actions
                # actions_i.append(a_t-1)
                if n_actions == 2:
                    actions.append(a_t * 2 - 1)
                else:
                    actions.append(a_t - 1)

                # create a one hot vector with the taken action and add to the action matrix
                action_blank = np.zeros(n_actions)
                action_blank[a_t] = 1
                actions_one_hot.append(action_blank)

                # create a one hot vector with the next state
                next_state_blank = np.zeros(2)
                next_state_blank[newState] = 1
                next_states_one_hot.append(next_state_blank)

                # calculation of the reward
                cum_reward += reward * gamma_cum
                gamma_cum = gamma_cum * gamma

                # states_i.append(np.append(np.append(state,action),theta))
                states.append(feats[state, :])
                next_states.append(np.array(newState - state))
                feat_diff.append(feats[newState, :] - feats[state, :])
                state = newState

                timesteps_i += 1

            if n % 10 == 0:
                print("Done: ", n / nb_episodes * 100)
                pass

            # states.append(states_i)
            # next_states.append(next_states_i)
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
            "next_states_one_hot": next_states_one_hot,
            "actions": actions,
            "paths": paths,
            "feat_diff": feat_diff,
        }
        return stats

    return sampling_fn


def collectData(
    agent, env, minTheta=5, maxTheta=15, bins=7, episode_count=5, timesteps=400
):
    # input and target for gaussian process
    input = None
    target = None

    # for theta in np.linspace(minTheta,maxTheta,bins):
    #     print("Simulation using: ", theta)

    #     env.setParams(theta)

    x, y = runEnv(
        env,
        episode_count=episode_count,
        timestep=timesteps,
        policy=agent,
        grid=True,
        theta_int=maxTheta,
    )

    # # stack horizontally x and theta, i.e. add theta as dimension of the input
    # x = np.hstack((x,np.ones((x.shape[0],1))*theta))

    # if input is None:
    #     input = x
    #     target = y

    # else:
    #     input = np.vstack((input,x))
    #     target = np.vstack((target,y))

    agent.store_data(x, y)


def make_pi(policy, sess, state_tf, action_size):
    def pi(state):
        probs = sess.run(policy, feed_dict={state_tf: state[np.newaxis, :]})[0]
        a = np.random.choice(int(action_size), p=probs)
        return a

    return pi


def testModelPolicy(
    env,
    policy,
    eval_steps=4,
    gamma=1,
    render=False,
    checkpoint_file="tf_checkpoint/general/model.ckpt",
    restore_variables=False,
    save_variables=True,
    logdir=None,
    log=False,
    overwrite_log=False,
    theta=5,
    use_gp_env=False,
    gp_env=None,
    **kwargs
):
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
    small_vel = 0
    obs_size = 2
    state_tf = tf.placeholder(tf.float32, (None, obs_size), name="states")
    policy_tf, _ = policy(state_tf)
    n_actions = 2

    # Start TF session
    with U.single_threaded_session() as sess:
        # to save variables
        saver = tf.train.Saver()

        # initialize all
        if restore_variables:
            # Add ops to save and restore all the variables.
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_file))
        else:
            init = tf.global_variables_initializer()
            sess.run(init)

        # make sure all variables are initialized
        sess.run(tf.assert_variables_initialized())
        pi = make_pi(policy_tf, sess, state_tf, n_actions)

        for n in range(10):
            paths.append(list())
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
            while not done:

                # Select action a_t according to current policy
                a_t = pi(state)
                env.render()

                newState, reward, done, info = env.step(a_t)

                # add to the buffer to remember
                # rewards_i.append(reward*gamma_cum)
                rewards.append(reward * gamma_cum)
                paths[n].append(newState)

                # works with two actions
                # actions_i.append(a_t-1)
                actions.append(a_t - 1)

                # create a one hot vector with the taken action and add to the action matrix
                action_blank = np.zeros(n_actions)
                action_blank[a_t] = 1
                # actions_i_one_hot.append(action_blank)
                actions_one_hot.append(action_blank)

                # calculation of the reward
                cum_reward += reward * gamma_cum
                gamma_cum = gamma_cum * gamma

                # states_i.append(np.append(np.append(state,action),theta))
                states_i.append(state)
                next_states_i.append(np.array(newState - state))
                state = newState

                timesteps_i += 1

                if info["goal_reached"]:
                    wins += 1
                    print(gamma_cum)
                if info["small_vel"]:
                    print("Small vel")
                    small_vel += 1

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
        }
    # print(stats)
    print(np.mean(stats["reward_list"]))
    return stats
