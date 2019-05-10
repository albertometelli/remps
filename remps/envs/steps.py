import math

import numba
import numpy as np
from numba import jit


@jit(nopython=True)
def MountainCarstep_small_vel(
    action, state, power, mass, steps, noise_vel, noise_pos, action_noise, max_steps
):

    min_position = -1.2
    max_position = 0.6
    max_speed = 0.1
    goal_position = 0.5
    # true if win, false otherwise
    # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

    position, velocity = state

    force = (action * power + action_noise) / mass

    # calculation of velocity based on the geometry of the environment
    velocity += force + math.cos(3 * position) * (-0.0025)

    # clip the velocity
    if velocity > max_speed:
        velocity = max_speed
    if velocity < -max_speed:
        velocity = -max_speed

    # update position
    position += velocity

    # add noise
    position += noise_pos
    velocity += noise_vel

    # clip position only if at left
    if position < min_position:
        position = min_position

    oldVelocity = velocity
    if position == min_position and velocity < 0:
        velocity = 0

    done = bool(position >= goal_position) or bool(steps >= max_steps)

    # if done the reward is 100 - reward_scaling*final_velocity
    # at each step the reward is - 1
    reward = 0
    if position >= goal_position:
        # reward = 200 - math.fabs(velocity)*self.reward_scaling
        goal_reached = True

    if steps >= max_steps:
        timeout = True

    # reward calculation
    reward, small_vel = reward_small_velocity(
        position, oldVelocity, action, goal_reached, timeout
    )

    # Reward n1: - action^2
    # reward -= math.pow(force,2)*0.1
    # Reward n2: - 1 for each timestep
    # reward = reward-1

    state = (position, velocity)
    return state, reward, done, goal_reached, small_vel


@jit(nopython=True)
def reward_small_velocity(position, velocity, force, goal_reached, timeout):

    min_position = -1.2

    # reward constants
    goal_reward_small_velocity = 100.0  # 1000
    timeout_penalization = 0
    small_velocity_threshold = 0.02
    left_wall_penalization = -100.0
    # goal reward
    if goal_reached and velocity < small_velocity_threshold:
        return goal_reward_small_velocity, True
    elif goal_reached:
        rew = (
            10.0
        )  # goal_reward_small_velocity - 100*goal_reward_small_velocity*math.fabs(velocity - small_velocity_threshold)
        return rew, False

    # penalize instability
    # if position == min_position:
    #     return left_wall_penalization*math.fabs(velocity), False

    # -1 for each timestep
    return -1, False  # + np.sin(3 * position)*.45#+.55


@jit(nopython=True)
def MountainCarstep_standard(
    action, state, power, mass, steps, noise_vel, noise_pos, max_steps
):

    min_position = -1.2
    max_position = 0.6
    max_speed = 0.1
    goal_position = 0.5
    # true if win, false otherwise
    # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

    position, velocity = state

    force = action * power / mass

    # calculation of velocity based on the geometry of the environment
    velocity += force + math.cos(3 * position) * (-0.0025)

    # clip the velocity
    if velocity > max_speed:
        velocity = max_speed
    if velocity < -max_speed:
        velocity = -max_speed

    # update position
    position += velocity

    # add noise
    position += noise_pos
    velocity += noise_vel

    # clip position
    if position < min_position:
        position = min_position

    oldVelocity = velocity
    if position == min_position and velocity < 0:
        velocity = 0

    done = bool(position >= goal_position) or bool(steps >= max_steps)

    # if done the reward is 100 - reward_scaling*final_velocity
    # at each step the reward is - 1
    reward = 0
    if position >= goal_position:
        # reward = 200 - math.fabs(velocity)*self.reward_scaling
        goal_reached = True

    if steps >= max_steps:
        timeout = True

    # reward calculation
    reward = standard_reward(position, oldVelocity, goal_reached, timeout)

    # Reward n1: - action^2
    # reward -= math.pow(force,2)*0.1
    # Reward n2: - 1 for each timestep
    # reward = reward-1

    state = (position, velocity)
    return state, reward, done, goal_reached, False


@jit(nopython=True)
def standard_reward(position, velocity, goal_reached, timeout):
    reward_scaling = 200
    # goal reward
    goal_distance = -8 * math.pow((position - 0.6), 2)
    if goal_reached:
        return math.exp(goal_distance)  # - math.pow(velocity,2)*reward_scaling

    return math.exp(goal_distance)


@jit(nopython=True)
def reward_small_action(
    goal_reached, goal_reward_small_action, max_steps, timeout, action, power
):
    if goal_reached:
        return goal_reward_small_action
    if timeout:
        # timeout is the same as doing always an action
        return -math.pow(power / 10, 8) * max_steps

    return -math.pow(action * power / 10, 8)


@jit(nopython=True)
def MountainCarstep_small_action(
    action, state, power, mass, steps, noise_vel, noise_pos, max_steps
):

    min_position = -1.2
    max_position = 0.6
    max_speed = 0.1
    goal_position = 0.5
    goal_reward_small_action = 500
    # true if win, false otherwise
    # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

    position, velocity = state

    force = action * power / mass

    # calculation of velocity based on the geometry of the environment
    velocity += force + math.cos(3 * position) * (-0.0025)

    # add gaussian noise on the velocity
    # velocity += np.random.normal(scale=noise_std)

    # clip the velocity
    if velocity > max_speed:
        velocity = max_speed
    if velocity < -max_speed:
        velocity = -max_speed
    # velocity = np.clip(velocity, -max_speed, max_speed)

    # update position and velocity
    position += velocity
    position += noise_pos
    velocity += noise_vel
    if position < min_position:
        position = min_position
    # position = np.clip(position, min_position, max_position)
    oldVelocity = velocity
    if position == min_position and velocity < 0:
        velocity = 0

    done = bool(position >= goal_position) or bool(steps >= max_steps)

    # if done the reward is 100 - reward_scaling*final_velocity
    # at each step the reward is - 1
    reward = 0
    if position >= goal_position:
        # reward = 200 - math.fabs(velocity)*self.reward_scaling
        goal_reached = True

    if steps >= max_steps:
        timeout = True

    # reward calculation
    reward = reward_small_action(
        done, goal_reward_small_action, max_steps, timeout, action, power
    )

    # Reward n1: - action^2
    # reward -= math.pow(force,2)*0.1
    # Reward n2: - 1 for each timestep
    # reward = reward-1

    state = (position, velocity)
    return state, reward, done, goal_reached, False


@jit(nopython=True)
def CartPoleStep(
    action,
    state,
    steps,
    force_mag,
    polemass_length,
    total_mass,
    masspole,
    length,
    gravity,
    tau,
    max_steps,
    x_noise,
    x_dot_noise,
    theta_noise,
    theta_dot_noise,
):
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4
    x, x_dot, theta, theta_dot = state
    force = force_mag if action == 1 else -force_mag
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    # add noise to the state
    x = x + tau * x_dot + x_noise
    x_dot = x_dot + tau * xacc + x_dot_noise
    theta = theta + tau * theta_dot + theta_noise
    theta_dot = theta_dot + tau * thetaacc + theta_dot_noise
    state = (x, x_dot, theta, theta_dot)
    done = (
        x < -x_threshold
        or x > x_threshold
        or theta < -theta_threshold_radians
        or theta > theta_threshold_radians
        or steps > max_steps
    )
    done = bool(done)

    if not done:
        reward = 1.0
    goal_reached = False
    if steps > max_steps:
        goal_reached = True

    return state, reward, done, goal_reached, False


@jit(nopython=True)
def CartPoleStepActionNoise(
    action,
    state,
    steps,
    force_mag,
    polemass_length,
    total_mass,
    masspole,
    length,
    gravity,
    tau,
    max_steps,
    action_noise,
    x_noise,
    x_dot_noise,
    theta_noise,
    theta_dot_noise,
):
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4
    x, x_dot, theta, theta_dot = state
    x_dot_old = x_dot
    force = force_mag if action == 1 else -force_mag
    # add action noise
    force = force + action_noise
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    # update first velocities and then other state components
    x_dot = x_dot + tau * xacc
    theta_dot = theta_dot + tau * thetaacc
    x = x + tau * x_dot + x_noise
    theta = theta + tau * theta_dot + theta_noise

    # add independent noise
    x_dot += x_dot_noise
    theta_dot += theta_dot_noise

    state = (x, x_dot, theta, theta_dot)
    done = (
        x < -x_threshold
        or x > x_threshold
        or theta < -theta_threshold_radians
        or theta > theta_threshold_radians
        or steps > max_steps
    )
    done = bool(done)
    if done and not steps > max_steps:
        reward = 0.0
    else:
        delta_vel = x_dot - x_dot_old
        reward = 10.0 - math.pow(force, 2) / 20.0 - 20.0 * (1 - math.cos(theta))
    goal_reached = False
    if steps > max_steps:
        goal_reached = True

    return state, reward, done, goal_reached, False
