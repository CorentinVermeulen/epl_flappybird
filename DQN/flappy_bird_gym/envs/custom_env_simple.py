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

class CustomEnvSimple(FlappyBirdEnvSimple):
    def __init__(self,
                 screen_size: Tuple[int, int] = (288, 512),
                 normalize_obs: bool = False,
                 pipe_gap: int = 100,
                 bird_color: str = "yellow",
                 pipe_color: str = "green",
                 background: Optional[str] = "day",
                 obs_var=None) -> None:
        super().__init__(screen_size=screen_size,
                         normalize_obs=normalize_obs,
                         pipe_gap=pipe_gap,
                         bird_color=bird_color,
                         pipe_color=pipe_color,
                         background=background)

        self._score = 0

        self.dict_obs = {}
        self.obs_var = obs_var
        self.rewards = {"alive": 0.1, "pass_pipe": 1, "dead": -1, 'score': 0}

        self.agent_vision = False
        self.reset()  # update self.observation_space with the new shape

    def __str__(self):
        str = (f"CustomEnvSimple(\n"
               f"\tScreen_size={self._screen_size} (normalized obs: {self._normalize_obs}"
               f"\tAction space: {self.action_space}\n"
               f"\tObservation space: {self.observation_space}\n"
               f"\tEnv obs variables: {self.obs_var}\n"
               f"\tEnv obs values: {self.reset()}\n"
               f"\tReward range: {self.rewards}\n"
               f")")
        return str

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
            'player_x': self._game.player_x,
            'player_y': player_y,
            'upper_pipe_y': upper_pipe_y,
            'lower_pipe_y': lower_pipe_y,
            'pipe_center_y': pipe_center,
            'pipe_center_x': low_pipe["x"],
            'v_dist': v_dist,
            'h_dist': h_dist,
            'player_vel_y': self._game.player_vel_y,
        }

        res = []
        if self.obs_var is None:
            for name in self.dict_obs:
                res.append(self.dict_obs[name])
        else:
            for name in self.obs_var:
                res.append(self.dict_obs[name])

        return np.array(res)

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

        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    info["QUIT"] = True
        except:
            pass

        return obs, reward, done, info

    def _define_reward(self, alive: bool):
        """
        I want to check:
        +0.2 or +score+0.1  if the bird is alive
        +0 or -1 if dead
        +1 or +0 if pass a pipe
        """
        reward = (0 + self.rewards['score'] * int(self._game.score)
                  + self.rewards['alive'] * int(alive)
                  + self.rewards['pass_pipe'] * int(self._game.score > self._score)
                  + self.rewards['dead'] * int(not alive)
                  )

        self._score = max(self._score, self._game.score)

        return reward

    def reset(self):
        """ Resets the environment (starts a new game). """
        self._score = 0
        obs = super().reset()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(len(obs),),
                                                dtype=np.float32)
        return obs

    def set_custom_render(self):
        self._renderer = CustomBirdRenderer(screen_size=self._screen_size,
                                            bird_color=self._bird_color,
                                            pipe_color=self._pipe_color,
                                            background=self._bg_type)
        self._renderer.game = self._game
        self._renderer.make_display()
        self.agent_vision = True

    def render(self, mode='human', stats="") -> None:
        """ Renders the next frame. """
        if self._renderer is None:
            self._renderer = FlappyBirdRenderer(screen_size=self._screen_size,
                                                bird_color=self._bird_color,
                                                pipe_color=self._pipe_color,
                                                background=self._bg_type)
            self._renderer.game = self._game
            self._renderer.make_display()

        if self.agent_vision:
            obs = {cle: valeur for cle, valeur in self.dict_obs.items() if cle in self.obs_var}
            self._renderer.draw_surface(show_score=True, stats=stats, obs=obs)

        else:
            self._renderer.draw_surface(show_score=True, stats=stats)

        self._renderer.update_display()

    def close(self):
        """ Closes the environment. """
        super().close()

    def update_game_logic_params(self, player_flap_acc , pipe_vel_x, player_acc_y):
        """
        Update the game logic parameters.
        player_flap_acc , pipe_vel_x, player_acc
        """
        self._game.update_params(player_flap_acc , pipe_vel_x, player_acc_y)
