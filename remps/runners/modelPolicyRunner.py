import copy
import os.path
import time
from collections import deque
from datetime import datetime
from multiprocessing import Event, Process, Queue
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug

import baselines.common.tf_util as U
from algo.fta import FTA
from algo.gradientDescent import Adam
from algo.policyGradientGPOMDP import GPOMDPOptimizer
from baselines.common.tf_util import GetFlat, SetFromFlat
from gp.gpManager import GpManager
from policy.MLPDiscrete import MLPDiscrete
from runners.envRunner import runEnv
from sampler.FTAparallelSampler import SamplingWorker
from utils.utils import make_session


def trainModelPolicy(
    env,
    policy,
    policy_optimizer,
    model_approximator,
    eval_steps=4,
    eval_freq=5,
    n_trajectories=20,
    iteration_number=2000,
    gamma=1,
    render=False,
    checkpoint_file="tf_checkpoint/general/model.ckpt",
    restore_variables=False,
    save_variables=True,
    logdir=None,
    log=False,
    start_from_iteration=0,
    overwrite_log=False,
    theta=5,
    use_gp_env=False,
    gp_env=None,
    render_train=False,
    model_optimizer=None,
    policy_opt_step=1,
    model_opt_step=1,
    **kwargs
):

    n_actions = env.action_space.n

    writer = tf.summary.FileWriter(logdir)

    # setup agent
    agent = FTA(
        policy,
        model_approximator,
        model_optimizer,
        policy_optimizer,
        env.observation_space.shape[0],
        n_actions,
        1,
        n_trajectories,
        writer,
        theta,
        0,
    )

    # create parallel samplers
    # Split work among workers
    num_processes = 4
    nb_episodes_per_worker = n_trajectories // num_processes

    inputQs = [Queue() for _ in range(num_processes)]
    outputQ = Queue()
    workers = [
        SamplingWorker(
            policy,
            env,
            nb_episodes_per_worker,
            inputQs[i],
            outputQ,
            n_actions,
            env.observation_space.shape[0],
        )
        for i in range(num_processes)
    ]

    # Run the Workers
    for w in workers:
        w.start()

    with U.single_threaded_session() as sess:

        # initialization with session
        agent.initialize(sess)

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

        print("Collecting Data")

        # first collect data
        collectData(
            agent, env, minTheta=5, maxTheta=15, bins=20, episode_count=1, timesteps=400
        )

        print("Data collected")

        # fit the model
        agent.fit()

        print("Model fitted")

        done = False

        reward_mean_to_plot = list()
        reward_std_to_plot = list()

        # set env params
        env.setParams(theta)

        get_parameters = U.GetFlat(agent.get_policy_params())

        for n in range(iteration_number):
            # we need to build three vectors that are the concatenations of state, actions and rewards for each trajectory
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

            start_time = time.time()

            policy_ws = get_parameters()

            # Run parallel sampling
            for i in range(num_processes):
                inputQs[i].put(("sample", policy_ws, theta))

            print("Workers started")
            # Collect results when ready
            for i in range(num_processes):
                pid, stats = outputQ.get()
                print(
                    "Collecting transition samples from Worker {}/{}".format(
                        i + 1, num_processes
                    )
                )
                states.extend(stats["states"])
                next_states.extend(stats["next_states"])
                rewards.extend(stats["rewards"])
                actions_one_hot.extend(stats["actions_one_hot"])
                actions.extend(stats["actions"])
                timesteps.extend(stats["timesteps"])
                reward_list.extend(stats["reward_list"])
                wins += stats["wins"]

            # print time
            print("Sampling requires {:.2f} seconds".format(time.time() - start_time))

            start_time = time.time()
            # We have finished to collect the gradient estimate, put everything in a matrix for padding
            # extract max length of episodes
            maxLength = np.max([len(l) for l in rewards])
            print("Max traj length", maxLength)

            # 1st dimension: #episodes
            # 2nd dimension: # timestep
            # 3rd dimension: characteristic dimension
            states_matrix = np.zeros(
                (len(states), maxLength, env.observation_space.shape[0])
            )
            next_states_matrix = np.zeros(
                (len(states), maxLength, env.observation_space.shape[0])
            )
            actions_matrix_one_hot = np.zeros((len(states), maxLength, n_actions))
            actions_matrix = np.zeros((len(states), maxLength))
            rewards_matrix = np.zeros((len(states), maxLength))
            mask = np.zeros((len(states), maxLength))

            for s, a_one_hot, a, s_prime, r, i in zip(
                states,
                actions_one_hot,
                actions,
                next_states,
                rewards,
                range(len(states)),
            ):
                states_matrix[i, 0 : len(s)] = s
                next_states_matrix[i, 0 : len(s_prime)] = s_prime
                rewards_matrix[i, 0 : len(r)] = r
                mask[i, 0 : len(r)] = np.ones_like(r)
                actions_matrix_one_hot[i, 0 : len(a_one_hot)] = a_one_hot
                actions_matrix[i, 0 : len(a)] = a

            # Finally apply the gradient
            traj_size = states_matrix.shape[0]

            print(
                "Data processing requires {:.2f} seconds".format(
                    time.time() - start_time
                )
            )

            # learning routine
            theta = agent.train(
                policy_opt_step,
                model_opt_step,
                mask.flatten(),
                states_matrix.reshape(
                    states_matrix.shape[0] * states_matrix.shape[1], -1
                ),
                next_states_matrix.reshape(
                    next_states_matrix.shape[0] * next_states_matrix.shape[1], -1
                ),
                actions_matrix_one_hot.reshape(
                    actions_matrix_one_hot.shape[0] * actions_matrix_one_hot.shape[1],
                    -1,
                ),
                actions_matrix.reshape(
                    actions_matrix.shape[0] * actions_matrix.shape[1], -1
                ),
                rewards_matrix.flatten(),
                theta,
                timesteps,
            )

            env.setParams(theta)

            if n % 5 == 0:
                # print statistics
                print("Training steps: ", n)
                print("Number of wins: ", wins)
                print("Percentage of wins: ", (wins / n_trajectories) * 100)
                print("Average reward: ", np.mean(reward_list))

                # fit gp
                # collectData(gpState, env, policy, theta, theta, 1)
                # gpState.fit()

            # show learned model and policy
            if ((n + 1) % eval_freq) == 0:

                # for plotting
                eval_rewards = []

                for i in range(eval_steps):

                    print("Evaluating...")
                    state = env.reset()

                    done = False

                    # gamma_cum is gamma^t
                    gamma_cum = 1
                    cum_reward = 0

                    t = 0
                    # here starts an episode
                    while not done:

                        if render:
                            env.render()

                        # sample one action at random
                        action = agent.pi(state[np.newaxis, :], log=log)

                        # observe the next state, reward etc
                        newState, reward, done, info = env.step(action)

                        cum_reward += reward * gamma_cum
                        gamma_cum = gamma * gamma_cum

                        state = newState

                        if done:
                            break

                        t = t + 1

                    eval_rewards.append(cum_reward)

                print("Average reward", np.mean(eval_rewards))

                reward_mean_to_plot.append(np.mean(eval_rewards))
                reward_std_to_plot.append(np.std(eval_rewards))

                # save variables
                if save_variables:
                    save_path = saver.save(sess, checkpoint_file)
                    print("Steps: ", n)
                    print("Model saved in path: %s" % save_path)

        # Close the env
        env.close()

        # save variables
        if save_variables:
            save_path = saver.save(sess, checkpoint_file)
            print("Model saved in path: %s" % save_path)


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

    agent.storeData(x, y)
