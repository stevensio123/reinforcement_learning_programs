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
                "agent": spaces.Box(
                    np.array([0, 0]),
                    np.array([y_size - 1, x_size - 1]),
                    shape=(2,),
                    dtype=int,
                )
            }
        )

        # To start agent & target location out of bounds initially
        self._agent_location = np.array([-1, -1], dtype=int)
        self._target_location = np.array([-1, -1], dtype=int)

        # Set up action space and movement of each action
        self.action_space = spaces.Discrete(9)
        self._action_to_direction = {
            Action.UP.value: np.array([-1, 0]),
            Action.DOWN.value: np.array([1, 0]),
            Action.LEFT.value: np.array([0, -1]),
            Action.RIGHT.value: np.array([0, 1]),
            Action.LEFT_UP.value: np.array([-1, -1]),
            Action.RIGHT_UP.value: np.array([-1, 1]),
            Action.LEFT_DOWN.value: np.array([1, -1]),
            Action.RIGHT_DOWN.value: np.array([1, 1]),
            Action.STAY.value: np.array([0, 0]),
        }

        # assert render_mode is None or render_mode is self.metadata["render_modes"]
        self.render_mode = render_mode

        # Only used when human is chosen from render_modes else stays as None
        self.window = None
        self.clock = None

    # Array to show valid actions to take in the current state
    def valid_action_space(self, agent_loc):
        state = agent_loc
        mask = np.ones(9, dtype=np.int8)
        if state[1] < 1:
            mask[[2, 4, 6]] = 0
        elif state[1] >= self.x_size - 1:
            mask[[3, 5, 7]] = 0
        if state[0] < 1:
            mask[[0, 4, 5]] = 0
        elif state[1] >= self.y_size - 1:
            mask[[1, 6, 7]] = 0

        return mask

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"action_mask": self.valid_action_space(self._agent_location)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Agent and target always starts at the same location so no need for np.random
        self._agent_location = np.array([3, 0])
        self._target_location = np.array([3, 7])

        obv = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obv, info

    def step(self, action):
        direction = self._action_to_direction[action]

        # For the effects of being in certain x-axis that causes increase in y-axis
        env_influence = np.array([0, 0])
        if self._agent_location[1] in [3, 4, 5, 8]:
            env_influence = np.array([-1, 0])
        elif self._agent_location[1] in [6, 7]:
            env_influence = np.array([-2, 0])

        # Move agent as follows based off action and env influence
        self._agent_location = np.clip(
            self._agent_location + env_influence + direction,
            [0, 0],
            [self.y_size - 1, self.x_size - 1],
        )

        obv = self._get_obs()
        info = self._get_info()

        # Check if agent has reached target location
        terminated = np.array_equal(self._agent_location, self._target_location)

        # Taken from reward given in exercise
        reward = 0 if terminated else -1

        if self.render_mode == "human":
            self._render_frame()

        return obv, reward, terminated, False, info

    def arrow_flipper(self, arrow, direction):
        match direction:
            case 1: # Down
                arrow[0][1] += self.pix_square_size[1]
            case 2: # Left
                arrow[1][0] -= self.pix_square_size[0] / 2
                arrow[1][1] -= self.pix_square_size[1] / 2


        return arrow

    def render(self):
        if self.render_mode == "rgb_array" or "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Set default rectangle size
        self.pix_square_size = (
            self.window_size / self.x_size,
            self.window_size / self.y_size,
        )

        # Draw target and agent
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self.pix_square_size
                * self._target_location[
                    ::-1
                ],  # -1 is to reverse the list to fit (x,y) coordinate format
                self.pix_square_size,
            ),
        )

        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                self.pix_square_size * self._agent_location[::-1],
                self.pix_square_size,
            ),
        )

        # Draw arrow NOTE: Current_location is default pointing up
        current_location = [
            [(self.pix_square_size[0] * (self._agent_location[1] + 1)) - self.pix_square_size[0]/2, self.pix_square_size[1] * self._agent_location[0]], # Top point
            [(self.pix_square_size[0] * (self._agent_location[1] + 1)), self.pix_square_size[1] * (self._agent_location[0] + 1) - self.pix_square_size[1]/2], # Right point
            [self.pix_square_size[0] * (self._agent_location[1]), (self.pix_square_size[1] * (self._agent_location[0] + 1)) - self.pix_square_size[1]/2], # Left point
            ]
        
        pygame.draw.polygon(
            canvas,
            (0, 0, 255),
            current_location
        )

        # Gridlines
        for x in range(self.x_size + 1):
            # draw.line func input (surface, colour, start_pos, end_pos,width)
            pygame.draw.line(
                canvas,
                0,
                (self.pix_square_size[0] * x, 0),
                (self.pix_square_size[0] * x, self.window_size),
                width=3,
            )
        for y in range(self.y_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0 ,self.pix_square_size[1] * y),
                (self.window_size, self.pix_square_size[1] * y),
                width=3,
            )

        if self.render_mode == "human":
            # Copy onto display and allow pygame to automatically run internal functions to prevent freezing
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # Allow the updates to happen at a pre-determined framerate
            self.clock.tick(self.metadata["render_fps"])

        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    # Just in case to close the env
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
