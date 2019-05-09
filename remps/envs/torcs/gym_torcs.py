import gym
from gym import spaces
import numpy as np

# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time

LAUNCH_TORCS = "torcs -T &"


class TorcsEnv:
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = (
        5
    )  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    def __init__(
        self, vision=False, throttle=False, gear_change=False, return_np=False
    ):
        # print("Init")
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.return_np = return_np

        self.initial_run = True

        ##print("launch torcs")
        # os.system('pkill torcs')
        # time.sleep(0.5)
        # os.system('torcs -nofuel -nodamage -nolaptime &')
        # time.sleep(0.5)
        # os.system('sh autostart.sh')
        # time.sleep(0.5)

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(
            p=3101, vision=self.vision
        )  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))

        high = np.array([1.0, np.inf, np.inf, np.inf, 1.0, np.inf, 1.0, np.inf, 255])
        low = np.array([0.0, -np.inf, -np.inf, -np.inf, 0.0, -np.inf, 0.0, -np.inf, 0])
        self.observation_space = spaces.Box(low=low, high=high)

    """
    Info contains:
    - backward
    - damage
    - progress_limit
    - out_of_track
    """

    def step(self, u):
        # print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs["steer"] = this_action["steer"]  # in [-1, 1]

        action_torcs["accel"] = this_action["accel"]

        action_torcs["brake"] = this_action["brake"]

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs["gear"] = this_action["gear"]
        else:
            #  Automatic Gear Change by Snakeoil is possible
            if client.S.d["speedX"] > 50:
                action_torcs["gear"] = 2
            if client.S.d["speedX"] > 80:
                action_torcs["gear"] = 3
            if client.S.d["speedX"] > 110:
                action_torcs["gear"] = 4
            if client.S.d["speedX"] > 140:
                action_torcs["gear"] = 5
            if client.S.d["speedX"] > 170:
                action_torcs["gear"] = 6

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observation(obs, return_np=self.return_np)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs["track"])
        sp = np.array(obs["speedX"])
        progress = sp * np.cos(obs["angle"])
        reward = progress

        info = {}

        # collision detection
        if obs["damage"] - obs_pre["damage"] > 0:
            reward = -1
            info["damage"] = 1

        # Termination judgement #########################
        episode_terminate = False
        # if track.min() < 0:  # Episode is terminated if the car is out of track
        #     reward = - 1
        #     episode_terminate = True
        #     client.R.d['meta'] = True
        #     info['out_of_track']= True

        if (
            self.terminal_judge_start < self.time_step
        ):  # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                episode_terminate = True
                client.R.d["meta"] = True
                info["progress_limit"] = True

        if np.cos(obs["angle"]) < 0:  # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d["meta"] = True
            info["backward"] = True

        if client.R.d["meta"] is True:  # Send a reset signal
            self.initial_run = False
            time = client.S.d["curLapTime"]
            dist = client.S.d["distFromStart"]
            totDist = client.S.d["totalDistFromStart"]
            distraced = client.S.d["distRaced"]
            lastLapTime = client.S.d["lastLapTime"]
            info.update(
                {
                    "time": time,
                    "distFromStart": dist,
                    "totalDistFromStart": totDist,
                    "distRaced": distraced,
                    "lastLapTime": lastLapTime,
                }
            )
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d["meta"], info

    def reset(self, relaunch=False):
        # print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(
            p=3101, vision=self.vision
        )  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observation(obs, return_np=self.return_np)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.system("pkill torcs")

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        # print("relaunch torcs")
        os.system("pkill torcs")
        time.sleep(0.5)
        os.system(LAUNCH_TORCS)
        time.sleep(0.5)
        # os.system('sh autostart.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {"steer": u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({"accel": u[1]})
            torcs_action.update({"brake": u[2]})

        if self.gear_change is True:  # gear change action is enabled
            torcs_action.update({"gear": u[3]})

        return torcs_action

    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec = obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0, 12286, 3):
            temp.append(image_vec[i])
            temp.append(image_vec[i + 1])
            temp.append(image_vec[i + 2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    """
    state is composed by:
    - focus : vector of 6 element with current focus
    - speed : x y z 3 element
    - rpm : engine 1 element
    - track : 6 sensors
    - wheelSpinVel: 4 element
    etc...
    total: 36 element 
    """

    def make_observation(self, raw_obs, return_np=False):
        if return_np:
            ob = np.concatenate(
                (
                    np.array(raw_obs["focus"], dtype=np.float32).reshape(-1) / 200,
                    np.array(raw_obs["speedX"], dtype=np.float32).reshape(-1)
                    / self.default_speed,
                    np.array(raw_obs["speedY"], dtype=np.float32).reshape(-1)
                    / self.default_speed,
                    np.array(raw_obs["speedZ"], dtype=np.float32).reshape(-1)
                    / self.default_speed,
                    np.array(raw_obs["rpm"], dtype=np.float32).reshape(-1),
                    np.array(raw_obs["track"], dtype=np.float32).reshape(-1) / 200.0,
                    np.array(raw_obs["wheelSpinVel"], dtype=np.float32).reshape(-1),
                    np.array(raw_obs["trackPos"], dtype=np.float32).reshape(-1),
                    np.array(raw_obs["angle"], dtype=np.float32).reshape(-1),
                    np.array(raw_obs["totalDistFromStart"], dtype=np.float32).reshape(
                        -1
                    ),
                    np.array(raw_obs["distFromStart"], dtype=np.float32).reshape(-1),
                )
            )
        else:
            names = [
                "focus",
                "speedX",
                "speedY",
                "speedZ",
                "opponents",
                "rpm",
                "track",
                "wheelSpinVel",
                "angle",
                "steer",
                "accel",
                "brake",
            ]
            ob = {}
            ob["focus"] = np.array(raw_obs["focus"], dtype=np.float32)
            ob["angle"] = np.array(raw_obs["angle"], dtype=np.float32)
            ob["accel"] = np.array(raw_obs["accel"], dtype=np.float32)
            ob["steer"] = np.array(raw_obs["steer"], dtype=np.float32)
            ob["brake"] = np.array(raw_obs["brake"], dtype=np.float32)
            ob["speedX"] = np.array(raw_obs["speedX"], dtype=np.float32)
            ob["speedZ"] = np.array(raw_obs["speedZ"], dtype=np.float32)
            ob["speedY"] = np.array(raw_obs["speedY"], dtype=np.float32)
            ob["rpm"] = np.array(raw_obs["rpm"], dtype=np.float32)
            ob["track"] = np.array(raw_obs["track"], dtype=np.float32)
            ob["trackPos"] = np.array(raw_obs["trackPos"], dtype=np.float32)
            ob["wheelSpinVel"] = np.array(raw_obs["wheelSpinVel"], dtype=np.float32)
        # print(obs.shape)
        return ob
