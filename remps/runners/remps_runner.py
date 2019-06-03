from multiprocessing import Queue

import baselines.common.tf_util as U
import numpy as np
import tensorflow as tf
from baselines import logger

from remps.algo.remps import REMPS, Projection
from remps.envs.confmdp import ConfMDP
from remps.model_approx.model_approximator import ModelApproximator
from remps.policy.policy import Policy
from remps.runners.envRunner import runEnv
from remps.sampler.parallel_sampler import SamplingWorker
from remps.utils.logging import timed


def train(
    env: ConfMDP,
    policy: Policy,
    model_approximator: ModelApproximator,
    eval_steps: int = 4,
    eval_freq: int = 5,
    n_trajectories: int = 20,
    iteration_number: int = 2000,
    gamma: float = 1,
    render=False,
    checkpoint_file: str = "tf_checkpoint/general/model.ckpt",
    restore_variables: bool = False,
    save_variables: bool = True,
    logdir: str = None,
    log: bool = False,
    omega=5,
    kappa: float = 1e-5,
    training_set_size: int = 500,
    normalize_data: bool = False,
    dual_reg: float = 0.0,
    policy_reg: float = 0.0,
    exact: bool = False,
    num_processes: int = 1,
    load_data: bool = True,
    **kwargs,
):
    """
    Runner for the REMPS algorithm.
    Setup logging, initialize agent, takes care of fitting or loading things.
    Executes the main training loop by managing workers
    :param env: Environment (Conf-MDP)
    :param policy: The agent policy
    :param model_approximator: the approximation of the model or the true model
    :param eval_steps: how many steps in order to perform evaluation
    :param eval_freq: the frequency of evaluation
    :param n_trajectories: number of trajectories to collect
    :param iteration_number: number of iterations of REMPS
    :param gamma: discount factor
    :param render: render or not episodes
    :param checkpoint_file: where to store checkpoints
    :param restore_variables: restore variables or not from checkpoint
    :param save_variables: save variables in checkpoint
    :param logdir: directory containing logs
    :param log: if true the agents logs the actions probability
    :param omega: initial environment parameters
    :param kappa: parameter of remps environment
    :param training_set_size: number of samples contained in the training set
    :param normalize_data: Whether to normalize data from the training set
    :param dual_reg: regularization on the dual
    :param policy_reg: regularization on the policy
    :param exact: whether the model approximation is exact or not
    :param num_processes: number of processing
    :param load_data: whether to load stored data
    :param kwargs:
    :return:
    """

    # setup logging
    writer = tf.summary.FileWriter(logdir)
    logger.configure(dir=logdir, format_strs=["stdout", "csv"])

    # setup agent
    agent = REMPS(
        policy=policy,
        model=model_approximator,
        env=env,
        kappa=kappa,
        projection_type=Projection.STATE_KERNEL,
        use_features=False,
        training_set_size=training_set_size,
        L2_reg_dual=dual_reg,
        L2_reg_loss=policy_reg,
        exact=exact,
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
            env,
            nb_episodes_per_worker,
            inputQs[i],
            outputQ,
            env.action_space.n,
            env.observation_space_size,
        )
        for i in range(num_processes)
    ]

    # Start the workers
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

        logger.log("Collecting Data", level=logger.INFO)

        # Collect data for model fitting
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

            # store data in the agent
            agent.store_data(x, y, normalize_data)

            logger.log("Data collected", logger.INFO)

        # fit the model
        agent.fit()

        logger.log("Model fitted", logger.INFO)

        # set configurable parameters
        env.set_params(omega)

        get_parameters = U.GetFlat(agent.get_policy_params())

        # -------------------------------------
        # --------- Training Loop -------------
        # -------------------------------------

        for n in range(iteration_number):
            states = list()
            next_states = list()
            rewards = list()
            actions_one_hot = list()
            actions = list()
            timesteps = list()
            paths = list()

            # statistics
            wins = 0
            small_vel = 0
            traj = 0
            confort_violation = 0
            reward_list = list()
            policy_ws = get_parameters()

            # Run parallel sampling:
            # for each worker send message sample with
            # policy weights and environment parameters
            for i in range(num_processes):
                inputQs[i].put(("sample", policy_ws, omega))

            # Collect results when ready
            with timed("sampling"):
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
            logger.log(f"Training steps: {n}", logger.INFO)
            logger.log(f"Number of wins: {wins}", logger.INFO)
            logger.log(f"Percentage of wins: {(wins/n_trajectories)*100}", logger.INFO)
            logger.log(f"Average reward: {np.mean(reward_list)}", logger.INFO)
            logger.log(f"Avg timesteps: {np.mean(timesteps)}")

            # learning routine
            with timed("training"):
                omega = agent.train(samples_data)

            # Configure environments with
            # parameters returned by the agent
            env.set_params(omega)

            # -------------------------------------
            # --------- Evaluation ----------------
            # -------------------------------------
            if ((n + 1) % eval_freq) == 0:

                # for plotting
                eval_rewards = []

                # evaluation loop
                for i in range(eval_steps):

                    logger.log("Evaluating...", logger.INFO)
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

                # save variables
                if save_variables:
                    save_path = saver.save(sess, checkpoint_file)
                    logger.log(f"Steps: {n}", logger.INFO)
                    logger.log(f"Model saved in path: {save_path}", logger.INFO)

        # Close the env
        env.close()

        # save variables
        if save_variables:
            save_path = saver.save(sess, checkpoint_file)
            logger.log(f"Model saved in path: {save_path}")

        # exit workers
        for i in range(num_processes):
            inputQs[i].put(("exit", None, None))
