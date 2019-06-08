import numpy as np


class TorcsAgent:
    """
    Bot for torcs environment.
    Taken from snakeoil bot: https://github.com/ugo-nama-kun/gym_torcs
    Only two actions:
    - Steer
    - Brake/acceleration

    Both actions are in the range [-1, 1]
    """

    def __init__(self, add_noise=False):
        self.count = 0
        self.add_noise = add_noise

    def pi(self, ob):
        target_speed = 100
        accel = ob["accel"]

        # Steer To Corner
        steer = ob["angle"] * 10 / np.pi
        # Steer To Center
        steer -= ob["trackPos"] * 0.10

        # Throttle Control
        if ob["speedX"] < target_speed - (steer * 50):
            accel += 0.01
        else:
            accel -= 0.01
        if ob["speedX"] < 10:
            accel += 1 / (ob["speedX"] + 0.1)

        # Traction Control System
        if (ob["wheelSpinVel"][2] + ob["wheelSpinVel"][3]) - (
            ob["wheelSpinVel"][0] + ob["wheelSpinVel"][1]
        ) > 5:
            accel -= 0.2

        # add noise coming from random normal
        if self.add_noise:
            steer += np.random.normal(0.0, 0.2)
            accel += np.random.normal(0.0, 0.2)

        # clip all
        action = np.array([np.clip(steer, -1, 1), np.clip(accel, -1, 1)])
        return action
