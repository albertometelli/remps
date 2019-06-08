import copy
import os
import time
import xml.etree.ElementTree as ET
from shutil import copyfile

import numpy as np

import remps.envs.torcs.snakeoil3_gym as snakeoil3
from gym import spaces
from remps.envs.confmdp import ConfMDP


class Torcs(ConfMDP):
    terminal_judge_start = 200  # Speed limit is applied after this step
    termination_limit_progress = (
        5
    )  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    def __init__(
        self,
        vision=False,
        throttle=True,
        gear_change=False,
        return_np=True,
        port=0,
        max_steps=1000,
        visual=False,
    ):
        # Setup torcs parameters
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.return_np = return_np
        self.initial_run = True
        self.time_step = 0
        self.max_steps = max_steps
        self.max_iters = 5
        self.cur_iter = 0
        # repeat action for action_repeat timesteps
        self.action_repeat = 1
        self.initial_reset = True
        # torcs property
        self.port = port
        self.visual = visual

        # number of params to configure
        self.n_configurable_parameters = 2
        self.configurable_parameters = np.ones(self.n_configurable_parameters)
        self.params_name = ["Rear Wing", "Front Wing"]
        self.params_attr = ["angle", "angle"]

        # check if conf file exists
        self.conf_file = f"/home/emanuele/opt/share/games/torcs/drivers/scr_server/1/default_{self.port}.xml"
        self.default_conf_file = (
            f"/home/emanuele/opt/share/games/torcs/drivers/scr_server/1/default.xml"
        )

        if not os.path.exists(self.conf_file):
            copyfile(self.default_conf_file, self.conf_file)

        self.tree = ET.parse(self.conf_file)
        self.root = self.tree.getroot()
        self.client = snakeoil3.Client(
            p=self.port,
            vision=self.vision,
            visual=visual,
            n_configurable_parameters=self.n_configurable_parameters,
        )  # Open new UDP in vtorcs
        self.client.MAX_STEPS = max_steps

        self.torcs_command = (
            "/home/emanuele/opt/bin/torcs -T -p " + str(self.port) + " &"
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        self.observation_space_dim = 29
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_space_size,)
        )
        self.action_space_size = 2

    def step(self, u):
        """
        Info contains:
        - backward
        - damage
        - progress_limit
        - out_of_track
        """
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        client.R.d["meta"] = False

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
        reward = 0

        info = {}

        episode_terminate = False

        for _ in range(self.action_repeat):
            # One-Step Dynamics Update
            # Apply the Agent's action into torcs
            client.respond_to_server()
            # Get the response of TORCS
            client.get_servers_input()

            # Get the current full-observation from torcs
            obs = client.S.d

            # Make an observation from a raw observation vector from TORCS
            self.observation = self.make_observation(obs, return_np=self.return_np)

            # Reward setting Here
            # direction-dependent positive reward
            track = np.array(obs["track"])
            sp = np.array(obs["speedX"])
            if np.isnan(sp):
                info["reset"] = True
            progress = sp * np.cos(obs["angle"])
            reward += progress

            if track.min() < 0:  # Episode is terminated if the car is out of track
                reward += -1000
                episode_terminate = True
                client.R.d["meta"] = True
                info["out_of_track"] = True

            if (
                np.cos(obs["angle"]) < 0
            ):  # Episode is terminated if the agent runs backward
                episode_terminate = True
                client.R.d["meta"] = True
                info["backward"] = True
                reward += -1000

            if (
                self.terminal_judge_start < self.time_step
            ):  # Episode is terminated if the agent is too slow
                if progress < self.termination_limit_progress:
                    episode_terminate = True
                    client.R.d["meta"] = True
                    info["slow"] = True
                    reward += -1000

        # Termination judgement
        if self.time_step >= self.max_steps:
            episode_terminate = True
            client.R.d["meta"] = True

        if episode_terminate is True:  # Send a reset signal
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

        return self.observation, reward, episode_terminate, info

    def get_state(self):
        self.client.get_servers_input()  # Get the initial input from torcs

        obs = self.client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observation(obs, return_np=self.return_np)
        return self.observation

    def reset(self, relaunch=False):

        self.time_step = 0
        self.cur_iter += 1

        if self.initial_reset is not True or self.cur_iter >= self.max_iters:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()
            time.sleep(1)
            # TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True or self.cur_iter >= self.max_iters:
                self.cur_iter = 0
                self.reset_torcs()

        self.client.shutdown()

        # setup new client
        self.client = snakeoil3.Client(
            p=self.port, vision=self.vision, visual=self.visual
        )  # Open new UDP in vtorcs
        self.client.MAX_STEPS = self.max_steps
        self.client.setParams(self.configurable_parameters)
        self.client.setup_connection(self.initial_reset)
        self.client.MAX_STEPS = self.max_steps

        self.client.get_servers_input()  # Get the initial input from torcs

        obs = self.client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observation(obs, return_np=self.return_np)

        self.last_u = None

        self.initial_reset = False
        return self.observation

    def close(self):
        self.client.R.d["exit"] = True
        self.client.respond_to_server()
        # time.sleep(1)
        self.client.shutdown()

    def reset_torcs(self):
        self.time_step = 0
        self.client.shutdown()
        self.client = snakeoil3.Client(
            p=self.port, vision=self.vision
        )  # Open new UDP in vtorcs
        self.client.MAX_STEPS = self.max_steps

    def agent_to_torcs(self, u):
        torcs_action = {"steer": u[0]}
        torcs_action.update({"accel": np.clip(u[1], 0, 1)})
        torcs_action.update({"brake": -np.clip(u[1], -1, 0)})
        torcs_action.update({"gear": 1})

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

    def get_params_bounds(self) -> np.array:
        min_params = np.zeros((self.n_configurable_parameters, 1))
        max_params = np.ones((self.n_configurable_parameters, 1))
        return np.hstack((min_params, max_params))

    @property
    def observation_space_size(self) -> int:
        return self.observation_space_dim

    def get_params(self) -> np.array:
        return self.configurable_parameters

    def set_params(self, *arg):
        self.configurable_parameters = arg[0]
        self.client.setParams(*arg)

        for p, p_name, p_attr in zip(
            self.configurable_parameters, self.params_name, self.params_attr
        ):
            for elem in self.root:
                # print(elem.attrib)
                if p_name in elem.attrib["name"]:
                    # print("found")
                    for child in elem:
                        if p_attr in child.attrib["name"]:
                            min_val = float(child.attrib["min"])
                            max_val = float(child.attrib["max"])
                            val = (max_val - min_val) * p + min_val
                            child.set("val", str(val))
                            break
                    break
        self.tree.write(self.conf_file)

    def make_observation(self, raw_obs, return_np=False):
        """
        state is composed by:
        - focus : vector of 5 element with current focus (not used)
        - speed : x y z 3 element
        - rpm : engine 1 element
        - track : 6 sensors
        - wheelSpinVel: 4 element
        etc...
        total: 29 element
        """
        if return_np:
            ob = np.concatenate(
                (  # np.array(raw_obs['focus'], dtype=np.float32).reshape(-1),
                    np.array(raw_obs["speedX"], dtype=np.float32).reshape(-1) / 50.0,
                    np.array(raw_obs["speedY"], dtype=np.float32).reshape(-1) / 50.0,
                    np.array(raw_obs["speedZ"], dtype=np.float32).reshape(-1) / 50.0,
                    np.array(raw_obs["rpm"], dtype=np.float32).reshape(-1) / 10000.0,
                    np.array(raw_obs["track"], dtype=np.float32).reshape(-1) / 200.0,
                    np.array(raw_obs["wheelSpinVel"], dtype=np.float32).reshape(-1),
                    np.array(raw_obs["trackPos"], dtype=np.float32).reshape(-1),
                    np.array(raw_obs["angle"], dtype=np.float32).reshape(-1) / 3.1416
                    # np.array(raw_obs['totalDistFromStart'], dtype=np.float32).reshape(-1),
                    # np.array(raw_obs['distFromStart'], dtype=np.float32).reshape(-1)
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
