import os.path
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug

from algo.pgReinforce import ReinforceOptimizer
from policy.MLPDiscrete import MLPDiscrete


def train(
    env,
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
):
    """Run GPOMDP optimizer over env using policy
    
    Parameters
    -----
    env: Environment

    policy: policy network, returning probabilities for each action

    sess: tensorflow session

    eval_steps: How many steps of evaluation to perform

    eval_freq: Evaluate every eval_freq steps

    iteration_number: how many steps of training to perform

    gamma: discount factor

    render: If True render evaluation

    log: If True log evaluation action probabilities

    checkpoint_file: where to save the variables

    save_variables: If True save variables

    logdir: Where to save tensorflow logs

    restore_variables: If True restore variables using checkpoint file

    n_trajectories: How many trajectories needed to estimate the gradient
    """

    n_actions = env.action_space.n

    now = datetime.now()

    if logdir is None:
        logdir = "tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"

    # do not overwrite log files
    if os.path.isdir(logdir):
        logdir = logdir[:-1] + now.strftime("%Y%m%d-%H%M%S") + "/"

    writer = tf.summary.FileWriter(logdir)

    optimizer = ReinforceOptimizer(
        policy.get_policy_network(),
        n_trajectories,
        policy.state,
        n_actions,
        writer,
        sess,
    )

    # to save variables
    saver = tf.train.Saver()

    # initialize all
    if restore_variables:
        # Add ops to save and restore all the variables.
        saver.restore(sess, checkpoint_file)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    # make sure all variables are initialized
    sess.run(tf.assert_variables_initialized())

    done = False

    reward_mean_to_plot = list()
    reward_std_to_plot = list()

    for n in range(iteration_number):
        # we need to build three vectors that are the concatenations of state, actions and rewards for each trajectory
        states = list()
        rewards = list()
        actions = list()
        timesteps = list()
        mask = None

        # statistics
        wins = 0
        reward_list = list()

        for i in range(n_trajectories):
            rewards_i = list()
            states_i = list()
            actions_i = list()
            mask_i = list()

            state = env.reset()
            states_i.append(state)
            done = False

            # gamma_cum is gamma^t
            gamma_cum = 1
            cum_reward = 0

            timesteps_i = 0

            # here starts an episode
            while not done:

                # sample one action at random
                action = policy.pi(state[np.newaxis, :], log=False)

                # observe the next state, reward etc
                newState, reward, done, info = env.step(action)

                if info:
                    wins += 1

                # add to the buffer to remember
                rewards_i.append(reward * gamma_cum)

                # create a one hot vector with the taken action and add to the action matrix
                action_blank = np.zeros(n_actions)
                action_blank[action] = 1
                actions_i.append(action_blank)

                # calculation of the reward
                cum_reward += reward * gamma_cum
                gamma_cum = gamma_cum * gamma

                state = newState

                if not done:
                    # the last state is useless, do not add it
                    states_i.append(newState)
                    timesteps_i += 1

            # the first axis is the number of trajectories
            states.append(states_i)
            actions.append(actions_i)
            rewards.append(rewards_i)
            timesteps.append(timesteps_i)

            reward_list.append(cum_reward)

        # We have finished to collect the gradient estimate, put everything in a matrix for padding
        # extract max length of episodes
        maxLength = np.max([len(l) for l in rewards])

        # 1st dimension: #episodes
        # 2nd dimension: # timestep
        # 3rd dimension: characteristic dimension
        states_matrix = np.zeros(
            (len(states), maxLength, env.observation_space.shape[0])
        )
        actions_matrix = np.zeros((len(states), maxLength, n_actions))
        rewards_matrix = np.zeros((len(states), maxLength))
        mask = np.zeros((len(states), maxLength))

        for s, a, r, i in zip(states, actions, rewards, range(len(states))):
            states_matrix[i, 0 : len(s)] = s
            actions_matrix[i, 0 : len(a)] = a
            rewards_matrix[i, 0 : len(r)] = r
            mask[i, 0 : len(r)] = np.ones_like(r)

        # Finally apply the gradient
        traj_size = actions_matrix.shape[0]
        optimizer.apply_grads(
            actions_matrix.reshape(
                actions_matrix.shape[0] * actions_matrix.shape[1], -1
            ),
            states_matrix.reshape(states_matrix.shape[0] * states_matrix.shape[1], -1),
            rewards_matrix.flatten(),
            mask.flatten(),
            timesteps,
        )

        if n % 5 == 0:
            # print statistics
            print("Training steps: ", n)
            print("Number of wins: ", wins)
            print("Percentage of wins: ", (wins / n_trajectories) * 100)
            print("Average reward: ", np.mean(reward_list))

        # show learned strategy
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
                print("Model saved in path: %s" % save_path)

    # Close the env
    env.close()


def test(
    env,
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
):
    """
    Restore the policy variables contained in checkpoint_file and run the policy for evaluation steps
    
    Parameters
    -----
    env: Environment

    policy: policy network, returning probabilities for each action

    sess: tensorflow session

    eval_steps: How many steps of evaluation to perform

    eval_freq: Evaluate every eval_freq steps

    iteration_number: how many steps of training to perform

    gamma: discount factor

    render: If True render evaluation

    log: If True log evaluation action probabilities

    checkpoint_file: where to save the variables

    save_variables: If True save variables

    logdir: Where to save tensorflow logs

    restore_variables: If True restore variables using checkpoint file

    n_trajectories: How many trajectories needed to estimate the gradient
    """
    n_actions = env.action_space.n

    # to restore variables
    saver = tf.train.Saver()

    # initialize all
    if restore_variables:
        # Add ops to save and restore all the variables.
        saver.restore(sess, checkpoint_file)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    # make sure all variables are initialized
    sess.run(tf.assert_variables_initialized())

    # for plotting
    eval_rewards = []

    for i in range(eval_steps):

        print("Episode Started")
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

        print("Timesteps: ", t)
        print("Reward: ", cum_reward)
