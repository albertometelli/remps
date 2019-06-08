import os
import os.path
from multiprocessing import Process

import numpy as np
import tensorflow as tf

import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from remps.envs.torcs.torcs import Torcs
from remps.policy.torcs_agent import TorcsAgent
from remps.utils.utils import get_default_tf_dtype


class TorcsSampler(Process):
    def __init__(
        self,
        policy,
        inputQ,
        outputQ,
        n_actions,
        obs_size,
        id,
        total_n_samples,
        n_params,
    ):
        # Invoke parent constructor BEFORE doing anything!!
        Process.__init__(self)
        self.dtype = get_default_tf_dtype()
        self.env = Torcs(port=id)
        self.state_tf = tf.placeholder(
            self.dtype, (None, self.env.observation_space_size), name="states"
        )
        self.action_tf = tf.placeholder(
            self.dtype, (None, self.env.action_space_size), name="actions"
        )
        self.policy_tf = None
        self.policy = policy
        self.inputQ = inputQ
        self.outputQ = outputQ
        self.n_actions = n_actions
        self.id = id
        self.total_n_samples = total_n_samples
        self.n_params = n_params

    def run(self):
        """Override Process.run()"""
        set_global_seeds(os.getpid())
        # set name of the policy
        self.policy.name = str(os.getpid()) + "policy"
        # create the policy
        self.policy_tf, _ = self.policy(self.state_tf, self.action_tf)

        # Start TF session
        with U.single_threaded_session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            pi = make_pi(self.policy._pi, sess, self.state_tf)
            # Build the sampling logic fn
            sampling_fn = make_sampling_fn(
                pi=pi,
                id=self.id,
                total_n_samples=self.total_n_samples,
                n_params=self.n_params,
            )
            inputs, targets, avg_reward, returns = sampling_fn()
            self.outputQ.put((os.getpid(), (inputs, targets, avg_reward, returns)))


def make_pi(policy, sess, state_tf):
    def pi(state):
        ac = sess.run(policy, feed_dict={state_tf: state})[0]
        return ac

    return pi


def make_sampling_fn(
    pi,
    id,
    use_scripted_policy=True,
    use_real_policy=False,
    n_samples_per_omega=1000,
    n_params=2,
    total_n_samples=1000,
):
    # Define the closure
    def sampling_fn():
        runs = 10
        env = Torcs(port=id)

        mult = 2 if use_scripted_policy and use_real_policy else 1
        inputs = np.zeros(
            (
                runs * n_samples_per_omega * mult,
                env.observation_space_size + env.action_space_size + n_params,
            )
        )
        targets = np.zeros(
            (runs * n_samples_per_omega * mult, env.observation_space_size)
        )
        i = 0

        returns = []
        avg_rew = []

        if use_real_policy:
            # Fit with current policy
            for n in range(runs):
                omegas = np.random.rand(n_params)
                env.set_params(omegas)
                state = env.reset()
                cum_rew = 0
                for t in range(n_samples_per_omega):
                    # sample one action from policy network or at random
                    action = pi(state.reshape(1, -1))

                    # save the current state action in the training set
                    inputs[i, :] = np.hstack((state, action, omegas))

                    # observe the next state, reward etc
                    new_state, reward, done, info = env.step(action)

                    cum_rew += reward

                    if info.get("reset", False):
                        state = env.reset()
                        break

                    targets[i, :] = np.matrix(new_state)

                    state = new_state

                    i += 1

                    if done:
                        state = env.reset()
                        print("Random", cum_rew)
                        cum_rew = 0
                print("************************************* Iteration: ", n)
        if use_scripted_policy:
            policy = TorcsAgent()
            # Fit with scripted policy
            for n in range(runs):
                omegas = np.random.rand(n_params)
                env.set_params(omegas)
                state = env.reset()
                cum_rew = 0
                for t in range(n_samples_per_omega):

                    action = policy.pi(
                        env.make_observation(env.client.S.d, return_np=False)
                    )

                    # save the current state action in the training set
                    inputs[i, :] = np.hstack((state, action, omegas))

                    # observe the next state, reward etc
                    new_state, reward, done, info = env.step(action)

                    cum_rew += reward

                    targets[i, :] = np.matrix(new_state)

                    state = new_state

                    i += 1

                    if done:
                        state = env.reset()
                        print("Return", cum_rew)
                        print("Average", cum_rew / t)
                        returns.append(cum_rew)
                        avg_rew.append(cum_rew / t)
                        cum_rew = 0
                print("************************************* Iteration: ", n)

        env.close()
        # subsampling
        ind = np.arange(0, np.shape(inputs)[0])
        selected_ind = np.random.choice(ind, size=total_n_samples, replace=True)
        inputs = inputs[selected_ind, :]
        targets = targets[selected_ind, :]

        print("Collected data points: ", inputs.shape)
        return inputs, targets, avg_rew, returns

    return sampling_fn
