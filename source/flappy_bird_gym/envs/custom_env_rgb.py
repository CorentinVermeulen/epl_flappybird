from typing import Dict, Tuple, Optional, Union

import gym
import numpy as np
import pygame

from flappy_bird_gym.envs.game_logic import FlappyBirdLogic
from flappy_bird_gym.envs.game_logic import PIPE_WIDTH, PIPE_HEIGHT
from flappy_bird_gym.envs.game_logic import PLAYER_WIDTH, PLAYER_HEIGHT
from flappy_bird_gym.envs.renderer import FlappyBirdRenderer
from flappy_bird_gym.envs.custom_renderer import CustomBirdRenderer
from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB


class CustomEnvRGB(FlappyBirdEnvRGB):
    def __init__(self,
                 screen_size: Tuple[int, int] = (288, 512),
                 pipe_gap: int = 100,
                 bird_color: str = "yellow",
                 pipe_color: str = "green",
                 background: Optional[str] = None) -> None:
        super().__init__(screen_size=(288, 512),
                         pipe_gap=100,
                         bird_color="yellow",
                         pipe_color="green",
                         background=None)

        self._score = 0
        self.game_params = {}
        self.rewards = {"alive": 0.1, "pass_pipe": 1, "dead": -1, 'score': 0}

        self.agent_vision = False
        self.reset()  # update self.observation_space with the new shape

    def __str__(self):
        str = (f"CustomEnvSimple(\n"
               f"\tScreen_size={self._screen_size}"
               f"\tAction space: {self.action_space}\n"
               f"\tObservation space: {self.observation_space}\n"
               f"\tEnv obs shape: {self.reset()[0].shape}\n"
               f"\tReward range: {self.rewards}\n"
               f")")
        return str

    def reset(self):
        """ Resets the environment (starts a new game). """
        self._score = 0
        self._game = FlappyBirdLogic(screen_size=self._screen_size,
                                     pipe_gap_size=self._pipe_gap)

        self._game.update_params(self.game_params)

        if self._renderer is not None:
            self._renderer.game = self._game

        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(len(self._get_observation()),),
                                                dtype=np.float32)
        return self._get_observation()

    def step(self,
             action: Union[FlappyBirdLogic.Actions, int],
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        """ Given an action, updates the game state.

        Args:
            action (Union[FlappyBirdLogic.Actions, int]): The action taken by
                the agent. Zero (0) means "do nothing" and one (1) means "flap".

        Returns:
            A tuple containing, respectively:

                * an observation (RGB-array representing the game's screen);
                * a reward (always 1);
                * a status report (`True` if the game is over and `False`
                  otherwise);
                * an info dictionary.
        """
        alive = self._game.update_state(action)
        obs = self._get_observation()

        reward = self._define_reward(alive)
        done = not alive

        info = {"score": self._game.score, "WIN": False, "QUIT": False}

        if self._game.score == 35:
            info["WIN"] = True
            reward += 100

        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    info["QUIT"] = True
        except:
            pass

        return obs, reward, done, info

    def _define_reward(self, alive: bool):
        reward = (0 + self.rewards['score'] * int(self._game.score)
                  + self.rewards['alive'] * int(alive)
                  + self.rewards['pass_pipe'] * int(self._game.score > self._score)
                  + self.rewards['dead'] * int(not alive)
                  )

        self._score = max(self._score, self._game.score)

        return reward

    def update_params(self, dic):
        self.game_params = dic
