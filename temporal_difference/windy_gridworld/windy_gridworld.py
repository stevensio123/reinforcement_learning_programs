from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import logging
import logging.config
from pathlib import Path
import numpy as np
import pygame

log_dir = Path(__file__).resolve.parent

formatter = logging.Formatter()
logging.config.dictconfig(
    {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(name)s - %(levelname)s | %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO",  # setting to INFO / DEBUG causes conflicts with displaying tqdm progress bars
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": f"{log_dir}/windy_gridworld.log",
                "formatter": "standard",
                "level": "DEBUG",
                "mode": "w",  # write or replace log file
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],  # logs to both console and file
        }
    }
)

logger = logging.getLogger(__name__)

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
                "agent": spaces.Box(np.array([0,0]), np.array([y_size - 1, x_size - 1]), shape=(2,), dtype=int),
                "target": spaces.Box(np.array([0,0]), np.array([y_size - 1, x_size - 1]), shape=(2,), dtype=int)
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

    def reset(self, seed=None, options=None):
        # Agent and target always starts at the same location so no need for np.random
        self._agent_location = np.array([3,0])
        self._target_location = np.array([3,7])

        logger.info("Environment succesfully resetted to agent start location: [3.0] & target location: [3,7]")

        if self.render_mode == "human":
            self._render_frame()
        
        return {"agent": self._agent_location, "target": self._target_location}
    
    def step(self, action):
        direction = self._action_to_direction[action]
        
        # For the effects of being in certain x-axis that causes increase in y-axis
        env_influence = np.array([0,0])
        if self._agent_location[1] in [3,4,5,8]:
            env_influence = np.array([1,0])
        elif self._agent_location[1] in [6,7]:
            env_influence = np.array([2,0])

        # Move agent as follows based off action and env influence
        self._agent_location = np.clip(
            self._agent_location + env_influence + direction, 0, [self.x_size - 1, self.y_size - 1]
        )

        # Check if agent has reached target location
        terminated = np.array_equal(self._agent_location, self._target_location)

        # Taken from reward given in exercise
        reward =  0 if terminated else -1

        logger.debug("Agent location: (%d, %d)", self._agent_location[1], self._agent_location[0])

        if self.render_mode == "human":
            self._render_frame()
        
        return {"agent": self._agent_location, "target": self._target_location}, reward, terminated, False