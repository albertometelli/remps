"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math

import numpy as np
from gym import spaces
from gym.utils import seeding

from remps.envs.confmdp import ConfMDP
from remps.envs.steps import CartPoleStepActionNoise as stepActionNoise


class CartPole(ConfMDP):
    def get_params_bounds(self) -> np.array:
        pass

    def get_params(self) -> np.array:
        pass

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, max_steps=100):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # noise
        self.action_noise_std = 1e-1
        self.x_range = 4.8
        self.theta_range = 180
        # the noise (3sigma) should be inside 10% of the range
        self.x_std = 1e-3  # (self.x_range/(3*1000))
        self.theta_std = 1e-3  # (self.theta_range/(3*1000))
        self.x_dot_std = 1e-3  # self.x_std/1e-3
        self.theta_dot_std = 1e-3  # self.theta_std/1e-3

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ]
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)
        self.observation_space_dim = 4
        self.n_actions = 2  # Left of Right

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.max_steps = max_steps
        self.steps = 0

    @property
    def observation_space_size(self):
        return self.observation_space_dim

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_params(self, omega, *args):
        if np.isscalar(omega):
            self.force_mag = omega
        else:
            # check the number of elements inside array is exactly 1
            assert omega.size == 1
            omega = omega.flatten()
            self.force_mag = omega[0]

    def step(self, action):

        self.steps += 1

        # sample noise
        noise_x = np.random.normal(scale=self.x_std)
        noise_x_dot = np.random.normal(scale=self.x_dot_std)
        noise_theta = np.random.normal(scale=self.theta_std)
        noise_theta_dot = np.random.normal(scale=self.theta_dot_std)
        action_noise_std = self.action_noise_std
        action_noise = np.random.normal(scale=action_noise_std)
        state, reward, done, goal_reached, _ = stepActionNoise(
            action,
            self.state,
            self.steps,
            self.force_mag,
            self.polemass_length,
            self.total_mass,
            self.masspole,
            self.length,
            self.gravity,
            self.tau,
            self.max_steps,
            action_noise,
            noise_x,
            noise_x_dot,
            noise_theta,
            noise_theta_dot,
        )
        self.state = state

        return (
            np.array(self.state),
            reward,
            done,
            {"goal_reached": goal_reached, "small_vel": False},
        )

    def getParams(self):
        return (
            self.state,
            self.steps,
            self.force_mag,
            self.polemass_length,
            self.total_mass,
            self.masspole,
            self.length,
            self.gravity,
            self.tau,
            self.max_steps,
        )

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.steps = 0
        return np.array(self.state)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
