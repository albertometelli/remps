import argparse
import os.path
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug

import gym
import remps.runners.GPOMDPModelRunner as modelRunner
import remps.runners.GPOMDPrunner as runner
import remps.runners.modelPolicyRunner as modelPolicyRunner
import remps.runners.p_remps_runner as premps_runner
import remps.runners.reps_runner as reps_runner
import remps.test.makePerformanceGrid as gridRunner

# log
from baselines import logger
from baselines.common.misc_util import set_global_seeds

# model optimizers
from remps.algo.modelGPOMDP import GPOMDPmodelOptimizer
from remps.algo.modelReinforce import ReinforceOptimizer
from remps.algo.offPGPOMDP import offPGPOMDPOptimizer as offPolicyGPOMDPOptimizer
from remps.algo.offPoffMModelGPOMDP import offPoffMGPOMDPmodelOptimizer
from remps.algo.pgReinforce import ReinforceOptimizer as reinforce
from remps.algo.policyGradientGPOMDP import GPOMDPOptimizer as gpomdp
from remps.envs.CartPoleEnv import CartPoleEnv
from remps.envs.Chain import NChainEnv
from remps.envs.MountainCarEnv import MountainCarConfEnv
from remps.envs.MountainCarEnvV2 import MountainCarEnv as mountainCarv2
from remps.envs.MountainCarGp import MountainCarEnv as MountainCarEnvGp
from remps.envs.puddleworld_conf_env import PuddleWorld
from remps.envs.torcs.gym_torcs import TorcsEnv

# Model approximators
from remps.gp.gpManager import GpManager
from remps.model_approx.CartPoleActionNoise import CartPoleModel as CartPoleActionNoise
from remps.model_approx.cartPoleModel import CartPoleModel
from remps.model_approx.ChainModel import ChainModel
from remps.model_approx.MountainCarActionNoise import MountainCarDummyApprox
from remps.model_approx.NNModel import NNModel
from remps.model_approx.PuddleWorldDummyApprox import PuddleWorldModel
from remps.policy.MLPDiscrete import MLPDiscrete
from remps.policy.OneParamPolicy import OneParam
from remps.test.testGp import testGp
from remps.utils.utils import boolean_flag, make_session

# Simulation parameters
EVALUATION_STEPS = 10
HIDDEN_LAYER_SIZE = 3
EVAL_FREQ = 50
N_TRAJECTORIES = 100
ITERATION_NUMBER = 500
MAX_STEPS = 500


def runExp(
    test,
    train,
    checkpoint_file,
    logdir,
    omega,
    noise_std,
    max_steps,
    train_model,
    hidden_layer_size,
    n_trajectories,
    gradient_estimator,
    reward_type,
    file_suffix,
    restore_variables,
    overwrite_log,
    n_actions,
    env_id,
    policy_optimizer,
    make_grid,
    test_gp,
    use_gp_env,
    model_gradient_estimator,
    use_gp_approx,
    train_model_policy,
    test_model_policy,
    use_remps,
    use_fta,
    use_premps,
    seed,
    exact,
    **kwargs
):

    policy_graph = None

    gp_env = None
    set_global_seeds(seed)

    policy = MLPGaussian()

    env_name = "torcs_env"
    env = TorcsEnv()

    # policy initialization
    if env_id != 3:
        policy = MLPDiscrete(
            env.observation_space.shape[0], env.action_space.n, hidden_layer_size
        )
    else:
        policy = OneParam()
    algo_name = ""
    if use_remps:
        algo_name = "REMPS"
    if use_premps:
        algo_name = "PREMPS"
    if use_fta:
        algo_name = "FTA"

    if train_model_policy:
        experiment_name = (
            algo_name
            + "/"
            + env_name
            + "-n-actions"
            + str(n_actions)
            + "-omega"
            + str(omega)
            + "-traj"
            + str(n_trajectories)
            + "-DualReg"
            + str(kwargs["dual_reg"])
            + "PolReg-"
            + str(kwargs["policy_reg"])
            + "TrainingSet"
            + str(kwargs["training_set_size"])
        )
    else:
        if not train_model:
            experiment_name = (
                env_name
                + "/rewardType"
                + str(reward_type)
                + "/GradientEstimator"
                + gradient_estimator
                + "/n-actions"
                + str(n_actions)
                + "/optimizer"
                + policy_optimizer
                + "/omega"
                + str(omega)
                + "-HiddenLayerSize"
                + str(hidden_layer_size)
                + "-traj"
                + str(n_trajectories)
            )
        else:
            experiment_name = (
                env_name
                + "/rewardType"
                + str(reward_type)
                + "/ModelGradientEstimator"
                + model_gradient_estimator
                + "/n-actions"
                + str(n_actions)
                + "/Policy-optimizer"
                + policy_optimizer
                + "/omega"
                + str(omega)
                + "-HiddenLayerSize"
                + str(hidden_layer_size)
                + "-traj"
                + str(n_trajectories)
            )

    if exact:
        experiment_name = experiment_name + "exact"
    experiment_name += str(seed)

    if file_suffix is not None:
        experiment_name = experiment_name + "-" + file_suffix

    # optimizer setting
    if gradient_estimator == "gpomdp":
        optimizer = gpomdp
    else:
        optimizer = reinforce

    if model_gradient_estimator == "gpomdp":
        print("Using gpomdp optimizer")
        model_optimizer = GPOMDPmodelOptimizer
    else:
        print("Using reinforce optimizer")
        model_optimizer = ReinforceOptimizer

    if logdir is None:
        if train_model:
            logdir = "tf_logs/model_logs/" + experiment_name + "/"
        else:
            if train_model_policy:
                logdir = (
                    "tf_logs/model_policy_logs/"
                    + experiment_name
                    + "eps-"
                    + str(kwargs["epsilon"])
                )
            else:
                logdir = "tf_logs/" + experiment_name + "/"

    now = datetime.now()

    # do not overwrite log files
    if os.path.isdir(logdir) and (not overwrite_log):
        logdir = logdir + "-" + now.strftime("%Y%m%d-%H%M%S") + "/"

    if checkpoint_file is None:
        if train_model_policy:
            experiment_name = (
                "model-policy/" + experiment_name + "eps-" + str(kwargs["epsilon"])
            )
        checkpoint_file = "tf_checkpoint/" + experiment_name + "/"

    if not restore_variables:
        # do not overwrite checkpoint files
        if os.path.isdir(checkpoint_file):
            checkpoint_file = checkpoint_file[:-1] + now.strftime("%Y%m%d-%H%M%S") + "/"
        else:
            os.makedirs(checkpoint_file)

    if train:
        checkpoint_file += "model.ckpt"

    print("Logs will be saved into: " + logdir)
    print("Checkpoints will be saved into: " + checkpoint_file)

    if train:
        runner.train(
            env,
            policy_graph,
            policy,
            sess,
            n_trajectories=n_trajectories,
            checkpoint_file=checkpoint_file,
            logdir=logdir,
            optimizer=optimizer,
            policy_optimizer=policy_optimizer,
            **kwargs
        )

    if test:
        runner.test(
            env,
            policy,
            checkpoint_file=checkpoint_file,
            optimizer=optimizer,
            restore_variables=restore_variables,
            **kwargs
        )

    if train_model:
        modelRunner.trainModel(
            env,
            policy_graph,
            policy,
            sess,
            n_trajectories=n_trajectories,
            checkpoint_file=checkpoint_file,
            logdir=logdir,
            theta=omega,
            restore_variables=restore_variables,
            use_gp_env=use_gp_env,
            gp_env=gp_env,
            model_optimizer=model_optimizer,
            **kwargs
        )

    if make_grid:
        print("Doing grid")
        gridRunner.makeGrid(
            policy,
            policy_graph,
            sess,
            env,
            checkpoint_file,
            5,
            50,
            200,
            n_trajectories,
            restore_variables,
            kwargs["render"],
            kwargs["title"],
        )

    if test_gp:
        print("Testing Gaussian Processes")
        testGp(
            env,
            policy=policy,
            checkpoint_file=checkpoint_file,
            policy_graph=policy_graph,
            sess=sess,
        )

    if train_model_policy:
        model_optimizer = offPoffMGPOMDPmodelOptimizer
        policy_optimizer = offPolicyGPOMDPOptimizer
        if use_gp_approx:
            model_approx = GaussianProcessManager(
                sess,
                env.observation_space.shape[0] + 1,
                env.observation_space.shape[0],
                1,
                scope="gp",
            )
            model_approx_to_optimize = GaussianProcessManager(
                sess,
                env.observation_space.shape[0] + 1,
                env.observation_space.shape[0],
                1,
                scope="gp_to_opt",
            )
        else:
            if env_id == 0:
                model_approx = NNModel(env.observation_space_size, 1, name=env_name)
            if env_id == 1:
                model_approx = CartPoleActionNoise()
                if not exact:
                    model_approx = NNModel(env.observation_space_size, 1, name=env_name)
            if env_id == 2:
                model_approx = PuddleWorldModel()
            if env_id == 3:
                model_approx = ChainModel()
        print("Training model and Policy...........")
        if use_remps:
            reps_runner.trainModelPolicy(
                env,
                policy,
                policy_optimizer,
                model_approx,
                n_trajectories=n_trajectories,
                checkpoint_file=checkpoint_file,
                logdir=logdir,
                omega=omega,
                restore_variables=restore_variables,
                use_gp_env=use_gp_env,
                gp_env=gp_env,
                model_optimizer=model_optimizer,
                exact=exact,
                **kwargs
            )
        if use_fta:
            modelPolicyRunner.trainModelPolicy(
                env,
                policy,
                policy_optimizer,
                model_approx,
                n_trajectories=n_trajectories,
                checkpoint_file=checkpoint_file,
                logdir=logdir,
                theta=omega,
                restore_variables=restore_variables,
                use_gp_env=use_gp_env,
                gp_env=gp_env,
                model_optimizer=model_optimizer,
                **kwargs
            )
        if use_premps:
            premps_runner.trainModelPolicy(
                env,
                policy,
                policy_optimizer,
                model_approx,
                n_trajectories=n_trajectories,
                checkpoint_file=checkpoint_file,
                logdir=logdir,
                omega=omega,
                restore_variables=restore_variables,
                use_gp_env=use_gp_env,
                gp_env=gp_env,
                model_optimizer=model_optimizer,
                **kwargs
            )

    if test_model_policy:
        reps_runner.testModelPolicy(
            env,
            policy,
            n_trajectories=n_trajectories,
            checkpoint_file=checkpoint_file,
            logdir=logdir,
            theta=omega,
            restore_variables=restore_variables,
            **kwargs
        )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    boolean_flag(parser, "render", help="Render evaluation", default=False)
    boolean_flag(parser, "render-train", help="Render training", default=False)
    boolean_flag(
        parser, "test", help="Run testing after training or only test", default=False
    )
    boolean_flag(parser, "train", default=False)
    boolean_flag(parser, "train-model", default=False)
    boolean_flag(parser, "use-remps", default=False)
    boolean_flag(parser, "use-premps", default=False)
    boolean_flag(parser, "use-fta", default=False)
    boolean_flag(parser, "test-gp", default=False)
    boolean_flag(parser, "train-model-policy", default=False)
    boolean_flag(parser, "test-model-policy", default=False)
    boolean_flag(parser, "restore-variables", default=False)
    boolean_flag(parser, "save-variables", default=True)
    boolean_flag(parser, "make-grid", default=False)
    boolean_flag(parser, "use-gp-env", default=False)
    # Use Gaussian Process for model approximation
    boolean_flag(parser, "use-gp-approx", default=False)
    parser.add_argument("--checkpoint-file", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eval-steps", type=int, default=EVALUATION_STEPS)
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--hidden-layer-size", type=int, default=HIDDEN_LAYER_SIZE)
    parser.add_argument(
        "--n-trajectories",
        help="Number of trajectories needed to estimate the gradient",
        type=int,
        default=N_TRAJECTORIES,
    )
    parser.add_argument(
        "--iteration-number",
        help="Number of training steps",
        type=int,
        default=ITERATION_NUMBER,
    )
    parser.add_argument(
        "--max-steps",
        help="Max number of steps of a trajectory",
        type=int,
        default=MAX_STEPS,
    )
    parser.add_argument(
        "--omega", help="Max speed of the car (MDP parameter)", type=float, default=9
    )
    parser.add_argument(
        "--noise-std",
        help="Noise standard deviation of the trasition function",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--gradient-estimator", help="gpomdp or reinforce", type=str, default="gpomdp"
    )
    parser.add_argument(
        "--model-gradient-estimator",
        help="gpomdp or reinforce",
        type=str,
        default="gpomdp",
    )
    parser.add_argument(
        "--start-from-iteration",
        help="Iteration number from which to start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--reward-type", help="Type of reward to use", type=int, default=0
    )
    boolean_flag(
        parser,
        "overwrite-log",
        help="If true overwrite logs in the folder",
        default=False,
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        help="suffix to add at the end of the file name",
        default=None,
    )
    parser.add_argument(
        "--n-actions", type=int, help="Number of actions in the environment", default=3
    )
    parser.add_argument(
        "--env-id", type=int, help="0 mountain car, 1 cartpole", default=0
    )
    parser.add_argument(
        "--policy-optimizer",
        type=str,
        help="gd (gradient descent), adam, adagrad, adadelta, rmsprop",
        default="gd",
    )
    parser.add_argument("--title", type=str, default="policy")
    parser.add_argument("--seed", type=int, default=1000)
    boolean_flag(parser, "exact", default=True)
    # REMPS
    parser.add_argument(
        "--epsilon", type=float, default=1e-3, help="Constraint on KL divergence"
    )
    parser.add_argument(
        "--dual-reg", type=float, default=0.0, help="Dual Regularization"
    )
    parser.add_argument(
        "--policy-reg", type=float, default=0.0, help="Policy Regularization"
    )
    parser.add_argument(
        "--training-set-size", type=int, default=500, help="Training set size"
    )
    boolean_flag(parser, "normalize-data", default=True)

    args = parser.parse_args()
    dict_args = vars(args)

    print("Running experiment with settings: ")
    print(dict_args)
    return dict_args


if __name__ == "__main__":
    args = parse_args()
    logger.configure()
    # Run actual script.
    runExp(**args)
