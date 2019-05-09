import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

# debug
from tensorflow.python import debug as tf_debug
import tensorflow.contrib.slim as slim
from algo.policyGradientGPOMDP import GPOMDPOptimizer
from policy.MLPDiscrete import MLPDiscrete
from gp.gpManager import GpManager
from runners.envRunner import runEnv
from datetime import datetime
from collections import deque
import os.path
from utils.utils import make_session
from algo.gradientDescent import Adam


def trainModel(
    env,
    policy_graph,
    policy,
    sess,
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
    **kwargs
):

    # GP definition
    # position, velocity, action, theta
    # lengthscales = np.array([0.542, 0.898, 16.4, 100])
    # signal_variance = 0.3**2
    # noise_variance = 1e-08
    # signal_variance = 4.6**2

    # lengthscales = np.array([7.48, 15.1, 146, 460])
    # signal_variance = 0.847**2
    # lengthscales = np.array([0.111, 0.083, 55.6, 250])
    # noise_variance = 1e-08

    # another
    # Position
    # Kernel 0.0502**2 * RBF(length_scale=[0.0855, 0.0579, 19.1, 49.3]) + WhiteKernel(noise_level=1.54e-07)
    # signal_variance = 0.0515**2
    # lengthscales= np.array([0.0855, 0.0579, 19.1, 49.3])
    # noise_variance = 1.54e-07
    signal_variance = 0.155 ** 2
    lengthscales = [0.164, 0.13, 1e03]
    noise_variance = 3.6e-07

    # Velocity
    # Kernel 0.0136**2 * RBF(length_scale=[0.068, 0.029, 0.02, 4.2]) + WhiteKernel(noise_level=1e-08)
    # signal_variance_vel = 0.0136**2
    # lengthscales_vel = [0.068, 0.029, 0.02, 4.2]
    # noise_variance_vel = 1e-08
    signal_variance_vel = 0.00978 ** 2
    lengthscales_vel = [0.341, 0.0652, 0.792]
    noise_variance_vel = 1e-08

    n_actions = env.action_space.n

    writer = tf.summary.FileWriter(logdir)

    model_graph = tf.Graph()

    model_sess = make_session(graph=model_graph)

    gpManager = GaussianProcessManager(
        model_graph,
        model_sess,
        env.observation_space.shape[0] + 1,
        env.observation_space.shape[0],
        1,
    )
    # restore variables
    with policy_graph.as_default():
        print("restoring variables")
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
        gpManager,
        env,
        minTheta=5,
        maxTheta=15,
        bins=20,
        episode_count=1,
        timesteps=400,
        policy=policy,
    )

    print("Data collected")

    # fit the gaussian process
    gpManager.fit()

    print("Model fitted")

    optimizer = model_optimizer(
        model_graph,
        gpManager.getProb(),
        n_trajectories,
        writer,
        model_sess,
        gpManager.theta,
    )

    # if use gp environment set this environment
    # transitions are sampled from gp
    if use_gp_env:
        env = gp_env
        env.setGp(gpManager)

    gd_optimizer = Adam(0.01, ascent=True)
    gd_optimizer.initialize(theta)

    done = False

    reward_mean_to_plot = list()
    reward_std_to_plot = list()
    # set env params
    env.setParams(theta)

    for n in range(iteration_number):
        # we need to build three vectors that are the concatenations of state, actions and rewards for each trajectory
        states = list()
        next_states = list()
        rewards = list()
        actions = list()
        timesteps = list()
        mask = None

        # statistics
        wins = 0
        reward_list = list()

        start_time = time.time()

        for i in range(n_trajectories):
            rewards_i = list()
            states_i = list()
            next_states_i = list()
            mask_i = list()

            state = env.reset()
            # states_i.append(state)
            done = False

            # gamma_cum is gamma^t
            gamma_cum = 1
            cum_reward = 0

            timesteps_i = 0

            # here starts an episode
            while not done:
                if render_train:
                    env.render()
                # sample one action at random
                action = policy.pi(state[np.newaxis, :], log=False)

                # observe the next state, reward etc
                newState, reward, done, info = env.step(action)

                if info:
                    wins += 1

                # if render:
                #     env.render()

                # add to the buffer to remember
                rewards_i.append(reward * gamma_cum)

                # calculation of the reward
                cum_reward += reward * gamma_cum
                gamma_cum = gamma_cum * gamma

                # states_i.append(np.append(np.append(state,action),theta))
                states_i.append(np.append(state, action - 1))
                next_states_i.append(np.array(newState - state))
                state = newState

                timesteps_i += 1

            # build the next state vector
            # leave the first state and keep only the first two components (position and velocity)
            # next_states_i = np.matrix(states_i)[1:,0:env.observation_space.shape[0]]
            # the first axis is the number of trajectories
            states.append(states_i)
            # add only the delta since GP predicts the delta state
            next_states.append(next_states_i)
            rewards.append(rewards_i)
            timesteps.append(timesteps_i)
            reward_list.append(cum_reward)

        # print time
        print("Sampling requires {:.2f} seconds".format(time.time() - start_time))

        start_time = time.time()
        # We have finished to collect the gradient estimate, put everything in a matrix for padding
        # extract max length of episodes
        maxLength = np.max([len(l) for l in rewards])

        # 1st dimension: #episodes
        # 2nd dimension: # timestep
        # 3rd dimension: characteristic dimension
        states_matrix = np.zeros(
            (len(states), maxLength, env.observation_space.shape[0] + 1)
        )
        next_states_matrix = np.zeros(
            (len(states), maxLength, env.observation_space.shape[0])
        )
        rewards_matrix = np.zeros((len(states), maxLength))
        mask = np.zeros((len(states), maxLength))

        for s, s_prime, r, i in zip(states, next_states, rewards, range(len(states))):
            states_matrix[i, 0 : len(s)] = s
            next_states_matrix[i, 0 : len(s_prime)] = s_prime
            rewards_matrix[i, 0 : len(r)] = r
            mask[i, 0 : len(r)] = np.ones_like(r)

        # Finally apply the gradient
        traj_size = states_matrix.shape[0]

        # print time
        print(
            "Data processing requires {:.2f} seconds".format(time.time() - start_time)
        )

        start_time = time.time()

        grad = optimizer.compute_grad(
            states_matrix.reshape(states_matrix.shape[0] * states_matrix.shape[1], -1),
            next_states_matrix.reshape(
                next_states_matrix.shape[0] * next_states_matrix.shape[1], -1
            ),
            rewards_matrix.flatten(),
            mask.flatten(),
            timesteps,
            np.matrix(theta),
            gpManager,
        )
        # print time
        print(
            "Gradient computation takes {:.2f} seconds".format(time.time() - start_time)
        )

        # update theta here
        print("Gradient: ", grad)

        # follow the gradient
        theta = gd_optimizer.update(grad)

        print("New value for theta: ", theta)

        # set env params
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
                    action = policy.pi(state[np.newaxis, :], log=log)

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
    gpManager,
    env,
    policy=None,
    minTheta=5,
    maxTheta=15,
    bins=7,
    episode_count=5,
    timesteps=400,
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
        policy=policy,
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

    gpManager.storeData(x, y)
