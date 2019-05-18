import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils.utils import plot


def makeGrid(
    policy,
    policy_graph,
    sess,
    env,
    checkpoint_file,
    min_value,
    max_value,
    bins,
    episodes,
    restore_variables,
    render,
    title,
):
    """
    Runs the policy for a grid of values and record the performances of that policy
    Parameters:
    - policy: network to be run
    - session
    """
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

    # dictionary of rewards:
    # key: parameter value
    # value: list of rewards
    rewards_dict = {}
    for theta in np.linspace(min_value, max_value, bins):

        print("Testing theta: ", theta)
        # set env param
        env.set_params(theta)

        reward_list = list()

        for i in range(episodes):
            state = np.array(env.reset())
            rewards = 0
            done = False
            while not done:

                if render:
                    env.render()

                # sample one action from policy network
                action = policy.pi(state[np.newaxis, :], log=False)

                # observe the next state, reward etc
                newState, reward, done, info = env.step(action)

                state = np.array(newState)

                rewards = rewards + reward

                if done:
                    break

            reward_list.append(rewards)

        rewards_dict[theta] = reward_list

    # print(rewards_dict)

    means = [np.mean(rewards_dict[x]) for x in np.linspace(min_value, max_value, bins)]

    variances = [
        np.std(rewards_dict[x]) for x in np.linspace(min_value, max_value, bins)
    ]

    x = np.linspace(min_value, max_value, bins)

    plot(x, means, variances, title)
