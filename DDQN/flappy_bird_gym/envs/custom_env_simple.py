
from typing import Dict, Tuple, Optional, Union

import gym
import numpy as np
import pygame

from flappy_bird_gym.envs.game_logic import FlappyBirdLogic
from flappy_bird_gym.envs.game_logic import PIPE_WIDTH, PIPE_HEIGHT
from flappy_bird_gym.envs.game_logic import PLAYER_WIDTH, PLAYER_HEIGHT
from flappy_bird_gym.envs.renderer import FlappyBirdRenderer
from flappy_bird_gym.envs.custom_renderer import CustomBirdRenderer
from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple



OBS_VAR = ['player_y','upper_pipe_y','lower_pipe_y','pipe_center_y','v_dist','h_dist','player_vel_y']

#OBS_VAR = ['player_x','player_y','pipe_center_x','pipe_center_y']


class CustomPleEnv(FlappyBirdEnvSimple):
    """
    Adaptation of FlappyBirdEnvSimple to a PyGame Learning Environment (PLE)
    """
    def __init__(self,
                 screen_size: Tuple[int, int] = (288, 512),
                 normalize_obs: bool = True,
                 pipe_gap: int = 100,
                 bird_color: str = "yellow",
                 pipe_color: str = "green",
                 background: Optional[str] = "day") -> None:
        super().__init__(screen_size, normalize_obs, pipe_gap, bird_color,
                         pipe_color, background)

    def reset_game(self):
        self._score = 0
        obs = super().reset()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(len(obs),),
                                                dtype=np.float32)

    def init(self):
        self.reset_game()

    def getActionSet(self):
        return [1,0]

    def getGameState(self):
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

        v_dist = (upper_pipe_y + lower_pipe_y) / 2 - (player_y
                                                      + PLAYER_HEIGHT / 2)

        if self._normalize_obs:
            h_dist /= self._screen_size[0]
            v_dist /= self._screen_size[1]

        state  ={
            'v_dist': v_dist,
            'h_dist': h_dist,
            'player_y': player_y,
            'player_x': self._game.player_x,
            'pipe_center_y': (upper_pipe_y + lower_pipe_y) / 2,
            'player_vel_y': self._game.player_vel_y,
        }
        return state

    def game_over(self):
        return not self.alive

    def score(self):
        return self._game.score
        pass

    def act(self, action):
        self.alive = self._game.update_state(action)
        obs = self._get_observation()

        reward = self._define_reward(self.alive)
        done = not self.alive

        info = {"score": self._game.score}

        if self._game.score == 50:
            info["WIN"] = True

        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    info["QUIT"] = True
        except:
            pass

        #return obs, reward, done, info
        return reward

    def _define_reward(self, alive: bool):
        reward = 0
        # If alive +0.1 else -1:
        if alive:
            reward += 0.1
        else:
            return -3

        if self._game.score > self._score:
            self._score = self._game.score
            reward += 1

        """ # If passed a pipe +1:
        if self._game.score > self._score:
            self._score = self._game.score
            if self.v_dist < self._game._pipe_gap_size / 2 :
                reward += 1
            reward += 1 
        """

        """ # If close to pipe gap center
        if self.v_dist < self._game._pipe_gap_size / 2:
            reward += 0.1
            """

        return reward