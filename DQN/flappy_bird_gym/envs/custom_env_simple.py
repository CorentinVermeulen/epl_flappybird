
from typing import Dict, Tuple, Optional, Union

import gym
import numpy as np
import pygame

from flappy_bird_gym.envs.game_logic import FlappyBirdLogic
from flappy_bird_gym.envs.game_logic import PIPE_WIDTH, PIPE_HEIGHT
from flappy_bird_gym.envs.game_logic import PLAYER_WIDTH, PLAYER_HEIGHT
from flappy_bird_gym.envs.renderer import FlappyBirdRenderer
from flappy_bird_gym.envs.agent_renderer import CustomBirdRenderer
from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple

class CustomEnvSimple(FlappyBirdEnvSimple):
    def __init__(self,
                 screen_size: Tuple[int, int] = (288, 512),
                 normalize_obs: bool = False,
                 pipe_gap: int = 100,
                 bird_color: str = "yellow",
                 pipe_color: str = "green",
                 background: Optional[str] = "day") -> None:
        super().__init__(screen_size=screen_size,
                         normalize_obs=normalize_obs,
                         pipe_gap=pipe_gap,
                         bird_color=bird_color,
                         pipe_color=pipe_color,
                         background=background)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(7,),
                                                dtype=np.float32)
        self._score = 0
        self.dict_obs = {}
        self.agent_vision = False

    def _get_observation(self):
        """

        Returns:
            A numpy array containing the current observation.
            player_y = player's y position;
            (upper_pipe_y + lower_pipe_y) / 2 = middle of the next hole;
            h_dist = horizontal distance to the next pipe;
            v_dist = difference between the player's y position and the next hole's y position.
            PLAYER_FLAP_ACC = Jump acceleration of the player.
            PIPE_VEL_X = Horizontal velocity of the pipes.
            PLAYER_ACC_Y = Vertical acceleration of the player.

        """
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

        self.dict_obs = {
            'player_y': player_y,
            'upper_pipe_y': upper_pipe_y,
            'lower_pipe_y': lower_pipe_y,
            'pipe_center': pipe_center,
            'v_dist': v_dist,
            'h_dist': h_dist,
            'player_vel_y': self._game.player_vel_y,
        }

        return np.array([
            player_y,
            upper_pipe_y,
            lower_pipe_y,
            pipe_center,
            v_dist,
            h_dist,
            self._game.player_vel_y,
        ])

    """     
            self._game.PLAYER_FLAP_ACC,
            self._game.PIPE_VEL_X,
            self._game.PLAYER_ACC_Y,
            self._game._pipe_gap_size
    """

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
        if self._game.score == 50:
            info["WIN"] = True

        return obs, reward, done, info

    def _define_reward(self, alive: bool):
        reward = 0
        # If alive +0.1 else -1:
        if alive:
            reward += 0.1
        else:
            return -1

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

    def reset(self):
        """ Resets the environment (starts a new game). """
        self._score = 0
        return super().reset()

    def set_custom_render(self):
        self._renderer = CustomBirdRenderer(screen_size=self._screen_size,
                                            bird_color=self._bird_color,
                                            pipe_color=self._pipe_color,
                                            background=self._bg_type)
        self._renderer.game = self._game
        self._renderer.make_display()
        self.agent_vision = True

    def render(self, mode='human', info="") -> None:
        """ Renders the next frame. """
        if self._renderer is None:
            self._renderer = FlappyBirdRenderer(screen_size=self._screen_size,
                                                bird_color=self._bird_color,
                                                pipe_color=self._pipe_color,
                                                background=self._bg_type)
            self._renderer.game = self._game
            self._renderer.make_display()

        if self.agent_vision:
            self._renderer.draw_surface(show_score=True, info=info, obs=self.dict_obs)

        else:
            self._renderer.draw_surface(show_score=True, info=info)

        self._renderer.update_display()

    def close(self):
        """ Closes the environment. """
        super().close()