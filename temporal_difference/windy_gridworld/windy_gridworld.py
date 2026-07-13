from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    LEFT_UP = 4
    RIGHT_UP = 5
    LEFT_DOWN = 6
    RIGHT_DOWN = 7
    STAY = 8

class WindyGridWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, x_size=10, y_size=7):
        self.x_size = x_size
        self.y_size = y_size
        self.window_size = 512

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(np.array([0,0], np.array[y_size - 1, x_size - 1], dtype=int)),
                "target": spaces.Box(np.array([0,0], np.array[y_size - 1, x_size - 1], dtype=int))
            }
        )

        # To start agent & target location out of bounds initially
        self._agent_location = np.array([-1,-1], dtype=int)
        self._target_location = np.array([-1,-1], dtype=int)

        # Set up action space and movement of each action
        self.action_space = spaces.Discrete(9)
        self._action_to_direction = {
            Action.UP.value: np.array([-1,0]),
            Action.DOWN.value: np.array([1,0]),
            Action.LEFT.value: np.array([0,-1]),
            Action.RIGHT.value: np.array([0,1]),
            Action.LEFT_UP.value: np.array([-1,-1]),
            Action.RIGHT_UP.value: np.array([-1,1]),
            Action.LEFT_DOWN.value: np.array([1,-1]),
            Action.RIGHT_DOWN.value: np.array([1,1]),
            Action.STAY.value: np.array([0,0]),
        }

        assert render_mode is None or render_mode is self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Only used when human is chosen from render_modes else stays as None
        self.window = None
        self.clock = None



