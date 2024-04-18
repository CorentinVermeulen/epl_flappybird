# MIT License
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

""" Implements the logic of the Flappy Bird game.

Some of the code in this module is an adaption of the code in the `FlapPyBird`
GitHub repository by `sourahbhv` (https://github.com/sourabhv/FlapPyBird),
released under the MIT license.
"""


import random
from enum import IntEnum
from itertools import cycle
from typing import Dict, Tuple, Union
from math import sin
import pygame

############################ Speed and Acceleration ############################
PIPE_VEL_X = -4        # Pipe velocity (-4)
OFFSET_SPAWNING_PIPES = 5 # Offset to spawn pipes (5)

PLAYER_MAX_VEL_Y = 10  # max vel along Y, max descend speed (10)
PLAYER_MIN_VEL_Y = -8  # min vel along Y, max ascend speed (-8)

PLAYER_ACC_Y = 1       # players downward acceleration (1)
PLAYER_VEL_ROT = 3     # angular speed (3)

PLAYER_FLAP_ACC = -9   # players speed on flapping (-9)
################################################################################


################################## Dimensions ##################################
PLAYER_WIDTH = 34
PLAYER_HEIGHT = 24

PIPE_WIDTH = 52
PIPE_HEIGHT = 320
PIPE_GAP_SIZE = 100

BASE_WIDTH = 336
BASE_HEIGHT = 112

BACKGROUND_WIDTH = 288
BACKGROUND_HEIGHT = 512
################################################################################
MAX_SCORE = 20

DEFINED_PIPES_VPOS = [125,  18,   0, 58,  73, 112,  33,   9,  7,  49,
                      114, 142, 100, 49,  53, 126,  89, 109, 97,  63,
                       24,  47,  53, 68, 141,  20,  41, 101, 80, 129,
                       44,  78,  55,  9,  84,  27,  84,  56, 39, 140,
                       20,  45,  64, 12,  34,   6, 117,  55, 62,   9,
                      125]

DEFINED_PIPES_VPOS_shuffle = [80, 55, 112, 45, 53, 34, 73, 142, 62, 101, 125, 97,
                              141, 58, 114, 55, 78, 140, 68, 7, 9, 63, 56, 20, 0,
                              84, 27, 126, 24, 109, 33, 18, 47, 9, 39, 9, 84, 20,
                              49, 117, 6, 129, 49, 44, 125, 53, 12, 41, 100, 89, 64]

DEFINED_PIPES_VPOS_2 = [70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0,
                      70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0,
                      70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0,
                      70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0,
                      70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0,
                      70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0,
                      70.0, 70.0]

class FlappyBirdLogic:
    """ Handles the logic of the Flappy Bird game.

    The implementation of this class is decoupled from the implementation of the
    game's graphics. This class implements the logical portion of the game.

    Args:
        screen_size (Tuple[int, int]): Tuple with the screen's width and height.
        pipe_gap_size (int): Space between a lower and an upper pipe.

    Attributes:
        player_x (int): The player's x position.
        player_y (int): The player's y position.
        base_x (int): The base/ground's x position.
        base_y (int): The base/ground's y position.
        score (int): Current score of the player.
        upper_pipes (List[Dict[str, int]): List with the upper pipes. Each pipe
            is represented by a dictionary containing two keys: "x" (the pipe's
            x position) and "y" (the pipe's y position).
        lower_pipes (List[Dict[str, int]): List with the lower pipes. Each pipe
            is represented by a dictionary containing two keys: "x" (the pipe's
            x position) and "y" (the pipe's y position).
        player_vel_y (int): The player's vertical velocity.
        player_rot (int): The player's rotation angle.
        last_action (Optional[FlappyBirdLogic.Actions]): The last action taken
            by the player. If `None`, the player hasn't taken any action yet.
        sound_cache (Optional[str]): Stores the name of the next sound to be
            played. If `None`, then no sound should be played.
        player_idx (int): Current index of the bird's animation cycle.
    """
    class Actions(IntEnum):
        """ Possible actions for the player to take. """
        IDLE, FLAP = 0, 1

    def __init__(self,
                 screen_size: Tuple[int, int],
                 pipe_gap_size: int = 100,
                 defined_pipes = DEFINED_PIPES_VPOS  ) -> None:
        self._screen_width = screen_size[0]
        self._screen_height = screen_size[1]

        self.player_x = int(self._screen_width * 0.2)
        self.player_y = int((self._screen_height - PLAYER_HEIGHT) / 2)

        self.base_x = 0
        self.base_y = self._screen_height * 0.79
        self._base_shift = BASE_WIDTH - BACKGROUND_WIDTH

        ######## Params to update ############################
        self.PIPE_VEL_X = PIPE_VEL_X  # Pipe velocity

        # Jump force
        self.jumpforce = PLAYER_FLAP_ACC
        self.PLAYER_FLAP_ACC = PLAYER_FLAP_ACC
        self.PLAYER_FLAP_ACC_VARIANCE = 0
        # Gravity
        self.gravity = PLAYER_ACC_Y
        self.PLAYER_ACC_Y = PLAYER_ACC_Y
        self.GRAVITY_SINUS_AMPLITUDE = 0
        self.GRAVITY_SINUS_PERIOD = 50

        # Pipe V position
        self.pipes_are_random = False
        self._defined_pipes = defined_pipes

        ######################################################
        self.score = 0
        self._pipe_gap_size = PIPE_GAP_SIZE
        self._pipe_nbr = 0

        # Generate 2 new pipes to add to upper_pipes and lower_pipes lists
        new_pipe1 = self._get_new_pipe()

        # List of upper pipes:
        self.upper_pipes = [
            {"x": self._screen_width + 200,
             "y": new_pipe1[0]["y"]},

        ]

        # List of lower pipes:
        self.lower_pipes = [
            {"x": self._screen_width + 200,
             "y": new_pipe1[1]["y"]},
        ]
        #{"x": self._screen_width + 200 + SPACE_BTW_PIPES * (self._screen_width / 2),"y": new_pipe2[1]["y"]},

        # Player's info:
        self.player_vel_y = -9  # player"s velocity along Y
        self.player_rot = 45  # player"s rotation

        self.last_action = None
        self.sound_cache = None

        self._player_flapped = False
        self.player_idx = 0
        self._player_idx_gen = cycle([0, 1, 2, 1])
        self._loop_iter = 0
        self._frame = 0

    def reset(self, params):
        self.__init__((self._screen_width, self._screen_height), self._pipe_gap_size, self._defined_pipes)
        self.update_params(params)

    def _update_gravity(self):
        self.gravity = self.PLAYER_ACC_Y + self.GRAVITY_SINUS_AMPLITUDE * sin(self._frame / self.GRAVITY_SINUS_PERIOD)
        return self.gravity

    def _update_jumpforce(self):
        self.jumpforce = self.PLAYER_FLAP_ACC * (1 + random.uniform(-1, 1) * self.PLAYER_FLAP_ACC_VARIANCE)
        return self.jumpforce

    def update_params(self, params) -> None:
        """ Updates the game's parameters.

        Args: params (Dict[str, int]): Dictionary containing the new values for the game's parameters.
        """
        for param_name, param_value in params.items():
            if hasattr(self, param_name):
                setattr(self, param_name, param_value)
            else:
                print(f"Warning: Attribute '{param_name}' does not exist in this object.")

    def switch_defined_pipes(self):
        self._defined_pipes = DEFINED_PIPES_VPOS_shuffle

    def _get_new_pipe(self) -> Dict[str, int]:
        """ Returns a randomly generated pipe. """
        # y of gap between upper and lower pipe
        if self.pipes_are_random:
            gap_y = random.randrange(0, int(self.base_y * 0.6 - self._pipe_gap_size))
        else:
            gap_y = self._defined_pipes[self._pipe_nbr]
            self._pipe_nbr +=1

        gap_y += int(self.base_y * 0.2)

        pipe_x = self._screen_width + 10
        return [{"x": pipe_x, "y": gap_y - PIPE_HEIGHT},  {"x": pipe_x, "y": gap_y + self._pipe_gap_size},]

    def check_crash(self) -> bool:
        """ Returns True if player collides with the ground (base)/sky or a pipe.
        """
        # if player crashes into ground
        if self.player_y + PLAYER_HEIGHT >= self.base_y - 1:
            return True
        if self.player_y < 0:
            return True
        else:
            player_rect = pygame.Rect(self.player_x, self.player_y,
                                      PLAYER_WIDTH, PLAYER_HEIGHT)

            for up_pipe, low_pipe in zip(self.upper_pipes, self.lower_pipes):
                # upper and lower pipe rects
                up_pipe_rect = pygame.Rect(up_pipe['x'], up_pipe['y'],
                                           PIPE_WIDTH, PIPE_HEIGHT)
                low_pipe_rect = pygame.Rect(low_pipe['x'], low_pipe['y'],
                                            PIPE_WIDTH, PIPE_HEIGHT)

                # check collision
                up_collide = player_rect.colliderect(up_pipe_rect)
                low_collide = player_rect.colliderect(low_pipe_rect)

                if up_collide or low_collide:
                    return True

        return False

    def update_state(self, action: Union[Actions, int]) -> bool:
        """ Given an action taken by the player, updates the game's state.

        Args:
            action (Union[FlappyBirdLogic.Actions, int]): The action taken by
                the player.

        Returns:
            `True` if the player is alive and `False` otherwise.
        """
        self.sound_cache = None
        if action == FlappyBirdLogic.Actions.FLAP:
            if self.player_y > -2 * PLAYER_HEIGHT:
                self.player_vel_y = self._update_jumpforce()
                self._player_flapped = True
                self.sound_cache = "wing"

        self.last_action = action
        if self.check_crash():
            self.sound_cache = "hit"
            return False

        # check for score
        player_mid_pos = self.player_x + PLAYER_WIDTH / 2
        for pipe in self.upper_pipes:
            pipe_mid_pos = pipe['x'] + PIPE_WIDTH / 2
            if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                self.score += 1
                self.sound_cache = "point"

        # If score = MAX_SCORE => winning
        if self.score == MAX_SCORE:
            return False

        # player_index base_x change
        if (self._loop_iter + 1) % 3 == 0:
            self.player_idx = next(self._player_idx_gen)

        self._loop_iter = (self._loop_iter + 1) % 30
        self.base_x = -((-self.base_x + 100) % self._base_shift)

        # rotate the player
        if self.player_rot > -90:
            self.player_rot -= PLAYER_VEL_ROT

        # player's movement
        if self.player_vel_y < PLAYER_MAX_VEL_Y and not self._player_flapped:
            self.player_vel_y += self._update_gravity()

        if self._player_flapped:
            self._player_flapped = False

            # more rotation to cover the threshold
            # (calculated in visible rotation)
            self.player_rot = 45

        self.player_y += min(self.player_vel_y,
                             self.base_y - self.player_y - PLAYER_HEIGHT)

        # move pipes to left
        for up_pipe, low_pipe in zip(self.upper_pipes, self.lower_pipes):
            up_pipe['x'] += self.PIPE_VEL_X
            low_pipe['x'] += self.PIPE_VEL_X

        # add new pipe when first pipe is about to touch left of screen
        if len(self.upper_pipes) > 0 and 0 < self.upper_pipes[0]['x'] < OFFSET_SPAWNING_PIPES:
            new_pipe = self._get_new_pipe()
            self.upper_pipes.append(new_pipe[0])
            self.lower_pipes.append(new_pipe[1])

            # remove first pipe if its out of the screen
        if (len(self.upper_pipes) > 0 and
                self.upper_pipes[0]['x'] < -PIPE_WIDTH):
            self.upper_pipes.pop(0)
            self.lower_pipes.pop(0)

        # Frame counter
        self._frame +=1
        return True