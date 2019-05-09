from rllab.algos.reps import REPS
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline


def run_task(*_):
    env = MountainCarEnv(height_bonus=0)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(5,),
    )

    # es = OUStrategy(env_spec=env.spec)

    baseline = LinearFeatureBaseline(env.spec)

    # qf = ContinuousMLPQFunction(env_spec=env.spec)

    algo = REPS(
        env=env,
        policy=policy,
        # es=es,
        # qf=qf,
        batch_size=200,
        max_path_length=1000,
        epoch_length=1000,
        min_pool_size=10000,
        n_epochs=1000,
        discount=1,
        scale_reward=1,
        # qf_learning_rate=1e-3,
        # policy_learning_rate=1e-4,
        baseline=baseline,
    )

    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
