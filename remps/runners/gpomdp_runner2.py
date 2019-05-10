import copy
import os.path
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from multiprocessing import Event, Process, Queue
from multiprocessing.pool import Pool

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug

import baselines.common.tf_util as U
from baselines import logger
from baselines.common import colorize
from baselines.common.tf_util import GetFlat, SetFromFlat
from remps.algo.gpomdp import GPOMDP
from remps.algo.gradientDescent import Adam
from remps.algo.remps import REPMS
from remps.envs.torcs.gym_torcs import TorcsEnv
from remps.policy.MLPDiscrete import MLPDiscrete
from remps.runners.envRunner import runEnv
from remps.sampler.fittingSampler import FittingSampler
from remps.sampler.trajectorySampler import SamplingWorker
from remps.utils.utils import get_default_tf_dtype

mpl.use("Agg")


def collectData(
    env,
    episode_count,
    bins,
    omega_max,
    omega_min,
    n_samples_per_omega,
    policy,
    grid,
    total_n_samples,
    n_params,
    initial_port,
):
    num_processes = 20
    nb_samples_per_worker = total_n_samples // num_processes

    inputQs = [Queue() for _ in range(num_processes)]
    outputQ = Queue()
    workers = [
        FittingSampler(
            policy,
            inputQs[i],
            outputQ,
            env.action_space_size,
            env.observation_space_size,
            i + initial_port,
            nb_samples_per_worker,
            n_params,
        )
        for i in range(num_processes)
    ]
    # Run the Workers
    for w in workers:
        w.start()

    X = None
    Y = None
    # Collect results when ready
    avg_rews_all = []
    returns_all = []
    for i in range(num_processes):
        _, pair = outputQ.get()
        x, y, avg_rew, returns = pair
        avg_rews_all.extend(avg_rew)
        returns_all.extend(returns)
        if X is None:
            X = x
            Y = y
        else:
            X = np.vstack((X, x))
            Y = np.vstack((Y, y))

    return X, Y, avg_rews_all, returns_all


def trainModelPolicy(
    env,
    policy,
    model_approximator,
    n_trajectories=20,
    iteration_number=2000,
    checkpoint_file="tf_checkpoint/general/model.ckpt",
    restore_variables=False,
    save_variables=True,
    logdir=None,
    epsilon=1e-5,
    training_set_size=100000,
    normalize_data=False,
    dual_reg=0.0,
    policy_reg=0.0,
    n_params=2,
    initial_port=3000,
    project_model=True,
    load_data=True,
    load_weights=True,
    load_policy=True,
    num_processes=20,
    **kwargs,
):
    n_actions = env.action_space_size

    writer = tf.summary.FileWriter(logdir)

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
    agent = GPOMDP(
        policy=policy,
        model=model_approximator,
        env=env,
        epsilon=epsilon,
        projection_type="joint",
        use_features=False,
        training_set_size=training_set_size,
        L2_reg_dual=dual_reg,
        L2_reg_loss=policy_reg,
        exact=True,
        logdir=logdir,
        project_model=project_model,
        load_weights=load_weights,
        load_policy=load_policy,
        n_trajectories=n_trajectories,
    )

    # create parallel samplers
    # Split work among workers
    n_steps = n_trajectories
    nb_episodes_per_worker = n_steps // num_processes

    inputQs = [Queue() for _ in range(num_processes)]
    outputQ = Queue()
    workers = [
        SamplingWorker(
            policy,
            nb_episodes_per_worker,
            inputQs[i],
            outputQ,
            n_actions,
            env.observation_space_size,
            initial_port + i,
        )
        for i in range(num_processes)
    ]

    # Run the Workers
    for w in workers:
        w.start()

    print("Collecting Data")

    # first collect data
    if not load_data:
        x, y, avg_rew, ret = collectData(
            env,
            episode_count=1,
            bins=200,
            omega_max=0,
            omega_min=1,
            n_samples_per_omega=100000,
            policy=policy,
            grid=False,
            total_n_samples=training_set_size,
            n_params=n_params,
            initial_port=initial_port + 1000,
        )

        print(f"Avg rew: {np.mean(avg_rew)}, Avg ret: {np.mean(ret)}")
        with open("stat.txt", "w") as f:
            f.write(f"Avg rew: {np.mean(avg_rew)}, Avg ret: {np.mean(ret)}")
        agent.storeData(x, y, normalize_data)

        states = x[:, : env.observation_space_size]
        actions = x[
            :,
            env.observation_space_size : env.observation_space_size
            + env.action_space_size,
        ]

        print("Data collected")

    print("Killing torcs")
    os.system("ps | grep torcs | awk '{print $1}' | xargs kill -9")

    with U.single_threaded_session() as sess:
        # omega = np.random.rand(n_params)
        omega = np.random.rand(n_params)
        # initialization with session
        agent.initialize(sess, writer, omega)

        # to save variables
        saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)

        init = tf.global_variables_initializer()
        sess.run(init)

        # make sure all variables are initialized
        sess.run(tf.assert_variables_initialized())

        # fit the policy
        if not load_policy:
            agent.fitPolicy(states, actions)
        else:
            theta = np.load("policy_weights.npy")
            U.SetFromFlat(agent.policy.trainable_vars, dtype=get_default_tf_dtype())(
                theta
            )
        print("Policy Fitted")

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
            times = list()
            speed = list()
            speedx = list()
            backward = list()
            outoftrack = list()
            distRaced = list()
            avg_rew = list()
            paths_full = {}
            paths_full["states"] = list()
            paths_full["actions"] = list()
            paths_full["next_states"] = list()
            paths_full["next_states_centred"] = list()
            paths_full["rewards"] = list()
            paths_full["actions_one_hot"] = list()
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
                    distRaced.extend(stats["distRaced"])
                    times.extend(stats["times"])
                    speed.extend(stats["speed"])
                    speedx.extend(stats["speedx"])
                    backward.extend(stats["backward"])
                    outoftrack.extend(stats["out_of_track"])
                    wins += stats["wins"]
                    small_vel += stats["small_vel"]
                    traj += stats["traj"]
                    confort_violation += stats["confort_violation"]
                    avg_rew.extend(stats["avg_rew"])
                    for key in paths_full:
                        paths_full[key].extend(stats["paths_full"][key])

            samples_data = {
                "actions": np.matrix(actions),
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
                "speed": speed,
                "times": times,
                "distRaced": distRaced,
                "speedx": speedx,
                "backward": backward,
                "out_of_track": outoftrack,
                "avg_rew": avg_rew,
                "paths_full": paths_full,
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

            if n % 10 == 0:
                print("Killing torcs")
                os.system("ps | grep torcs | awk '{print $1}' | xargs kill -9")

            if n % 20 == 0:
                # save variables
                if save_variables:
                    save_path = saver.save(sess, checkpoint_file, global_step=n)
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
    **kwargs,
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
