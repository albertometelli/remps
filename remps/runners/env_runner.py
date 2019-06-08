import numpy as np


def run_env(
    env,
    episode_count=100,
    n_samples_per_omega=100,
    policy=None,
    grid=False,
    omega_min=0,
    omega_max=10,
    bins=100,
    total_n_samples=500,
):
    """
    Simple runner, takes an environment, run a random policy and records everything
    """
    if not grid:
        inputs = np.zeros((bins * n_samples_per_omega, env.observation_space_size + 1))
        targets = np.zeros((bins * n_samples_per_omega, env.observation_space_size))
        i = 0
        for omega in np.linspace(omega_min, omega_max, bins):
            env.set_params(omega)
            state = np.array(env.reset())
            for t in range(n_samples_per_omega):
                # sample one action from policy network or at random
                if policy is None:
                    action = env.action_space.sample()
                else:
                    action = policy.pi(state[np.newaxis, :], log=False)

                if env.n_actions == 2:
                    action = action * 2 - 1
                else:
                    action = action - 1

                force = action * omega
                # save the current state action in the training set
                inputs[i, :] = np.hstack((state, force))

                # observe the next state, reward etc
                newState, reward, done, info = env.step(action)

                newState = np.array(newState)

                # compute the delta to be added in the target
                delta = np.matrix((newState - state))

                targets[i, :] = delta

                state = newState

                i += 1

                if done:
                    state = np.array(env.reset())

        env.close()
    else:
        low_pos, low_vel = env.low
        high_pos, high_vel = env.high

        # actions = np.random.randint(low=0, high=env.n_actions, size=timestep)# [1.0/env.n_actions]*env.n_actions)
        actions = np.random.uniform(
            low=-omega_max, high=omega_max, size=total_n_samples
        )
        positions = np.random.uniform(low=low_pos, high=high_pos, size=total_n_samples)
        velocities = np.random.uniform(low=low_vel, high=high_vel, size=total_n_samples)

        start_states = list(zip(positions, velocities, actions))
        inputs = np.matrix(start_states)
        next_states = list()

        action = 1

        for state in start_states:
            x, x_dot, a = state
            env.set_params(a)
            newState, reward, done, info = env._step(action, (x, x_dot))
            # append delta state
            next_states.append(newState - np.array([x, x_dot]))

        targets = np.matrix(next_states)
        return inputs, targets

        # # modify actions:
        # if env.n_actions == 3:
        #     inputs[:,2] = inputs[:,2] - 1
        # else:
        #     inputs[:,2] = 2*inputs[:,2] - 1

    # subsampling
    ind = np.arange(0, np.shape(inputs)[0])
    selected_ind = np.random.choice(ind, size=total_n_samples, replace=True)
    inputs = inputs[selected_ind, :]
    targets = targets[selected_ind, :]

    print("Collected data points: ", inputs.shape)
    return inputs, targets
