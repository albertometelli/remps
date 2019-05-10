import copy
import os.path
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from multiprocessing import Event, Process, Queue
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug

import baselines.common.tf_util as U
from baselines import logger
from baselines.common import colorize
from baselines.common.tf_util import GetFlat, SetFromFlat
from remps.algo.gradientDescent import Adam
from remps.algo.remps import REPMS
from remps.policy.MLPDiscrete import MLPDiscrete
from remps.runners.envRunner import runEnv
from remps.sampler.parallelSampler2 import SamplingWorker


def trainModelPolicy(
    env,
    policy,
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
    omega=5,
    epsilon=1e-5,
    training_set_size=500,
    normalize_data=False,
    dual_reg=0.0,
    policy_reg=0.0,
    exact=False,
    **kwargs
):
    n_actions = env.action_space.n

    writer = tf.summary.FileWriter(logdir)
    load_data = True

    logger.configure(dir=logdir, format_strs=["stdout", "csv"])

    @contextmanager
    def timed(msg):
        print(colorize(msg, color="red"))
        tstart = time.time()
        yield
        print(
            colorize(
                msg + " done in %.3f seconds" % (time.time() - tstart), color="red"
            )
        )

    # setup agent
    agent = REPMS(
        policy=policy,
        model=model_approximator,
        env=env,
        epsilon=epsilon,
        projection_type="joint",
        use_features=False,
        training_set_size=training_set_size,
        L2_reg_dual=dual_reg,
        L2_reg_loss=policy_reg,
        exact=exact,
    )

    # create parallel samplers
    # Split work among workers
    num_processes = 1
    n_steps = n_trajectories
    nb_episodes_per_worker = n_steps // num_processes

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
            env.observation_space_size,
        )
        for i in range(num_processes)
    ]

    # Run the Workers
    for w in workers:
        w.start()

    with U.single_threaded_session() as sess:

        # initialization with session
        agent.initialize(sess, writer, omega)

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
        if not load_data:
            x, y = runEnv(
                env,
                episode_count=1,
                bins=200,
                omega_max=30,
                omega_min=1,
                n_samples_per_omega=500,
                policy=agent,
                grid=True,
                total_n_samples=training_set_size,
            )

            agent.storeData(x, y, normalize_data)

            print("Data collected")

        # fit the model
        agent.fit()

        print("Model fitted")

        done = False

        reward_mean_to_plot = list()
        reward_std_to_plot = list()

        # set env params
        env.setParams(omega)

        get_parameters = U.GetFlat(agent.get_policy_params())

        for n in range(iteration_number):
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
            traj = 0
            confort_violation = 0
            reward_list = list()
            policy_ws = get_parameters()

            # Run parallel sampling
            for i in range(num_processes):
                inputQs[i].put(("sample", policy_ws, omega))

            with timed("sampling"):
                # Collect results when ready
                for i in range(num_processes):
                    _, stats = outputQ.get()
                    states.extend(stats["states"])
                    paths.extend(stats["paths"])
                    next_states.extend(stats["next_states"])
                    rewards.extend(stats["rewards"])
                    actions_one_hot.extend(stats["actions_one_hot"])
                    actions.extend(stats["actions"])
                    timesteps.extend(stats["timesteps"])
                    reward_list.extend(stats["reward_list"])
                    wins += stats["wins"]
                    small_vel += stats["small_vel"]
                    traj += stats["traj"]
                    confort_violation += stats["confort_violation"]

            samples_data = {
                "actions": np.matrix(actions).transpose(),
                "actions_one_hot": np.array(actions_one_hot),
                "observations": states,
                "paths": paths,
                "rewards": np.transpose(np.expand_dims(np.array(rewards), axis=0)),
                "reward_list": reward_list,
                "timesteps": timesteps,
                "wins": (wins / traj) * 100,
                "omega": omega,
                "traj": traj,
                "confort_violation": confort_violation,
            }

            # print statistics
            print("Training steps: ", n)
            print("Number of wins: ", wins)
            print("Percentage of wins: ", (wins / n_trajectories) * 100)
            print("Average reward: ", np.mean(reward_list))
            print("Avg timesteps: ", np.mean(timesteps))
            print("Win with small vel: ", small_vel)

            # learning routine
            with timed("training"):
                omega = agent.train(samples_data)

            env.setParams(omega)

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

        # exit workers
        for i in range(num_processes):
            inputQs[i].put(("exit", None, None))


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
