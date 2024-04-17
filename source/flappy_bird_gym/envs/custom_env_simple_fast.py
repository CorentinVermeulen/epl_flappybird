#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Implementation of a Flappy Bird OpenAI Gym environment that yields simple
numerical information about the game's state as observations.
"""

from typing import Dict, Tuple, Optional, Union

import gym
import numpy as np
import pygame

from flappy_bird_gym.envs.game_logic import FlappyBirdLogic
from flappy_bird_gym.envs.game_logic import PIPE_WIDTH, PIPE_HEIGHT
from flappy_bird_gym.envs.game_logic import PLAYER_WIDTH, PLAYER_HEIGHT
from flappy_bird_gym.envs.renderer import FlappyBirdRenderer


class FlappyBirdEnvSimpleFast(gym.Env):
    """ Flappy Bird Gym environment that yields simple observations.

    The observations yielded by this environment are simple numerical
    information about the game's state. Specifically, the observations are:

        * Horizontal distance to the next pipe;
        * Difference between the player's y position and the next hole's y
          position.

    The reward received by the agent in each step is equal to the score obtained
    by the agent in that step. A score point is obtained every time the bird
    passes a pipe.

    Args:
        screen_size (Tuple[int, int]): The screen's width and height.
        normalize_obs (bool): If `True`, the observations will be normalized
            before being returned.
        pipe_gap (int): Space between a lower and an upper pipe.
        bird_color (str): Color of the flappy bird. The currently available
            colors are "yellow", "blue" and "red".
        pipe_color (str): Color of the pipes. The currently available colors are
            "green" and "red".
        background (Optional[str]): Type of background image. The currently
            available types are "day" and "night". If `None`, no background will
            be drawn.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 screen_size: Tuple[int, int] = (288, 512),
                 normalize_obs: bool = True,
                 pipe_gap: int = 100,
                 obs_var=None) -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(2,),
                                                dtype=np.float32)
        self._screen_size = screen_size
        self._normalize_obs = normalize_obs
        self._pipe_gap = pipe_gap
        self._game = FlappyBirdLogic(screen_size=self._screen_size,
                                         pipe_gap_size=self._pipe_gap)
        self._bg_type = "day"

        self._score = 0
        self.game_params = {}
        self.obs_gravity = False
        self.obs_jumpforce = False
        self.rewards = {"alive": 0.1, "pass_pipe": 1, "dead": -1, 'score': 0}

        self.reset()  # update self.observation_space with the new shape

    def __str__(self):
        str = (f"CustomEnvSimple(\n"
               f"\tScreen_size={self._screen_size} (normalized obs: {self._normalize_obs}"
               f"\tAction space: {self.action_space}\n"
               f"\tObservation space: {self.observation_space}\n"
               f"\tEnv obs values: {len(self.reset())}\n"
               f"\tReward range: {self.rewards}\n"
               f"\tObs gravity: {self.obs_gravity}\n"
               f"\tObs jumpforce: {self.obs_jumpforce}\n"
               f"\tGame parameters: \n")
        for p in self.game_params:
            str += f"\t\t{p}: {self.game_params[p]}\n"
        str += ")"
        return str

    def reset(self):
        """ Resets the environment (starts a new game). """
        self._score = 0
        self._game.reset(self.game_params)


        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(len(self._get_observation()),),
                                                dtype=np.float32)
        return self._get_observation()

    def _get_observation(self):
        up_pipe = low_pipe = None
        h_dist = 0
        for up_pipe, low_pipe in zip(self._game.upper_pipes,
                                     self._game.lower_pipes):
            h_dist = (low_pipe["x"] + PIPE_WIDTH / 2
                      - (self._game.player_x - PLAYER_WIDTH / 2))
            h_dist += 3  # extra distance to compensate for the buggy hit-box
            if h_dist >= 0:
                break

        upper_pipe_y = up_pipe["y"] + PIPE_HEIGHT
        lower_pipe_y = low_pipe["y"]
        player_y = self._game.player_y

        pipe_center = (upper_pipe_y + lower_pipe_y) / 2
        v_dist = pipe_center - (player_y + PLAYER_HEIGHT / 2)

        if self._normalize_obs:
            h_dist /= self._screen_size[0]
            v_dist /= self._screen_size[1]

        res = [self._game.player_x , player_y , upper_pipe_y, lower_pipe_y, pipe_center, low_pipe["x"], v_dist, h_dist, self._game.player_vel_y]

        if self.obs_gravity:
            res.append(self._game.gravity)
        if self.obs_jumpforce:
            res.append(self._game.jumpforce)

        return np.array(res, dtype=np.float32)

    def step(self,
             action: Union[FlappyBirdLogic.Actions, int],
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """ Given an action, updates the game state.

        Args:
            action (Union[FlappyBirdLogic.Actions, int]): The action taken by
                the agent. Zero (0) means "do nothing" and one (1) means "flap".

        Returns:
            A tuple containing, respectively:

                * an observation (horizontal distance to the next pipe;
                  difference between the player's y position and the next hole's
                  y position);
                * a reward (always 1);
                * a status report (`True` if the game is over and `False`
                  otherwise);
                * an info dictionary.
        """
        alive = self._game.update_state(action)
        obs = self._get_observation()

        reward = self._define_reward(alive)
        done = not alive

        info = {"score": self._game.score}

        if self._game.score == 20:
            info["WIN"] = True
            reward += 100
            done=True

        return obs, reward, done, info

    def _define_reward(self, alive: bool):
        reward = (0 + self.rewards['score'] * int(self._game.score)
                  + self.rewards['alive'] * int(alive)
                  + self.rewards['pass_pipe'] * int(self._game.score > self._score)
                  + self.rewards['dead'] * int(not alive)
                  )

        self._score = max(self._score, self._game.score)

        return reward

    def close(self):
        """ Closes the environment. """
        super().close()

    def update_params(self, dic):
        self.game_params = dic