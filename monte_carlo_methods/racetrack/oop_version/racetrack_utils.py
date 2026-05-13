import numpy as np
import random
import logging
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)


class Racetrack:
    def __init__(self, racetrack):
        """
        class to represent state space of racetrack.
        state value is initialized to random integer between -5 and 1 (inclusive) for each state.
        """
        logger.info("creating racetrack...")
        # reverse and transpose racetrack to match coordinate system (x rows, y rows)
        racetrack_reverse = racetrack[::-1]
        self.racetrack = np.array([list(row) for row in racetrack_reverse]).T

        rng = np.random.default_rng()
        self.action_values = np.round(
            rng.uniform(-5, -2, size=(len(self.racetrack), len(self.racetrack[0]), 5, 5, 3, 3)), 3
        )  # NEW added rounding to 3 d.p for more accurate rounding

        self.start_coord_list = []
        self.terminal_coord_list = []
        for i in range(len(self.racetrack)):
            for j in range(len(self.racetrack[i])):
                if self.racetrack[i][j] == "E":
                    self.terminal_coord_list.append((i, j))
                    self.action_values[i][j] = 0
                if self.racetrack[i][j] == "S":
                    self.start_coord_list.append((i, j))

        self.x_terminal_loc = self.terminal_coord_list[0][0]
        self.y_terminal_locs = []
        for coord in self.terminal_coord_list:
            self.y_terminal_locs.append(coord[1])
        self.y_terminal_smallest_loc = min(self.y_terminal_locs)

        # create target_policy actions that is greedy to best action
        self.target_policy_dict = np.empty((len(racetrack[0]), len(racetrack), 5, 5), dtype=object)

    def get_action_value(self, state, action):
        x, y, vx, vy = state
        ax, ay = action
        # logger.debug("getting state value of state: %d, %d, %d, %d", x, y, vx, vy)
        return self.action_values[x, y, vx, vy, ax, ay]
        # equivalent to:
        # return self.action_values[x][y][vx][vy][ax][ay]


def get_next_state(Racetrack, state, action):
    # logger.debug("getting next state of {}".format(state))
    x, y, vx, vy = state
    vx += action[0]
    vy += action[1]
    x += vx
    y += vy
    if Racetrack.x_terminal_loc <= x and Racetrack.y_terminal_smallest_loc <= y:
        x = Racetrack.x_terminal_loc
        y = Racetrack.y_terminal_smallest_loc
    try:
        Racetrack.racetrack[x][y]
        # logger.debug("next state is %s", Racetrack.racetrack[x][y])
    except IndexError:  # out of bounds
        # logger.debug("next state is out of bounds.")
        new_coord = Racetrack.start_coord_list[
            np.random.randint(len(Racetrack.start_coord_list))
        ]
        return (new_coord[0], new_coord[1], 0, 0)
    if Racetrack.racetrack[x][y] == "#":  # crash
        # logger.debug("car crashed, starting over.")
        new_coord = Racetrack.start_coord_list[
            np.random.randint(len(Racetrack.start_coord_list))
        ]
        return (new_coord[0], new_coord[1], 0, 0)
    else:
        return (x, y, vx, vy)


def get_action_space(state):
    """
    Takes in state (tuple of x, y, vx, vy) and returns list of possible actions (acceleration) that can be taken from that state.
    Acceleration can be -1, 0, or 1 in both x and y, and velocity < 5
    """
    action_space = []
    accel = [-1, 0, 1]
    x, y, vx, vy = state
    for horizontal in accel:
        if 0 <= (vx + horizontal) < 5:
            for vertical in accel:
                if 0 < (vy + vertical) < 5 or (
                    (vx + horizontal) != 0 and 0 <= (vy + vertical) < 5
                ):
                    action_space.append([horizontal, vertical])
    return action_space


def get_policy(obj: Racetrack, epsilon=0.1):
    """
    takes in a Racetrack object and epsilon value,
    returns an array with the same shape as the racetrack state space,
    with the chosen action for each state according to the behavior policy as the elements' value.
    """
    action_values = obj.action_values

    # empty array of same shape as state space (x, y, vx, vy) with action as element value
    # use "object" type array to store lists as elements (because there are two actions)
    policy = np.empty((len(obj.racetrack), len(obj.racetrack[0]), 5, 5), dtype=object) 
    # logger.debug("generating policy with epsilon = %d", epsilon)
    for x in range(action_values.shape[0]):
        for y in range(action_values.shape[1]):
            for vx in range(action_values.shape[2]):
                for vy in range(action_values.shape[3]):
                    state = (x, y, vx, vy)
                    # choose optimal / random action:
                    action_space_ls = get_action_space(state)
                    if random.random() > epsilon:
                        # take optimal action according to current state values
                        action_idx = get_optimal_action(obj, state, action_space_ls)
                        policy[x][y][vx][vy] = action_space_ls[action_idx]
                    else:
                        # take random action
                        action_idx = np.random.randint(len(action_space_ls))
                        policy[x][y][vx][vy] = action_space_ls[action_idx]
    return policy

def get_optimal_action(Racetrack, state, action_space_ls, step_action=None):
    """
    takes in a StateSpace object and epsilon value,
    returns an array with the same shape as the state space,
    with the chosen action for each state according to the behavior policy as the elements.
    """
    action_values = Racetrack.action_values
    action_values = []
    for idx, action in enumerate(action_space_ls):
        action_value = Racetrack.get_action_value(state, action)
        action_values.append(action_value)
        if step_action == action: 
            """
            check if actual action taken in episode is equal to current action in the loop, 
            if so, store the action value and index of that action to compare with optimal action value later
            """
            policy_action_idx = idx
            policy_action_value = action_value
    # if multiple actions have the same action value, randomly pick one of them as the optimal action
    action_idx = random.choice(np.where(action_values == np.max(action_values))[0])
    if step_action != None:  
        if policy_action_value == np.max(action_values): # else left blank so if condition not satisfied, old action_idx is kept
            """
            if the action taken in the episode is one of the optimal actions, 
            then we later use this to update the policy to keep that action as optimal, 
            instead of randomly picking another optimal action.
            """
            action_idx = policy_action_idx
    # print(f"largest state value = {np.max(action_values)}")
    return action_idx


class Episode:
    """
    takes the Racetrack class as an input parameter, this is so we can access the start/terminal locs method.
    creates episode by following the policy until it reaches terminal state.
    episode is stored as a list of tuples of: (state, action) pairs.
    """

    def __init__(self, Racetrack, policy):
        """
        policy: a 4D array that has an action for each state (x, y, vx, vy)
        super().__init__(racetrack)
        """
        # start_loc: randomly chosen starting coordinate
        self.terminal_locs = Racetrack.terminal_coord_list  # Moved to init as useful
        self.episode = []
        self.policy = policy

    def generate(self, Racetrack, max_steps=100, start_pos=None):
        """
        method to create episode by following the policy until it reaches terminal state.
        """
        if start_pos == None:
            current_loc = Racetrack.start_coord_list[
                np.random.randint(len(Racetrack.start_coord_list))
            ]
        else:
            current_loc = start_pos
        current_state = (current_loc[0], current_loc[1], 0, 0)
        self.steps = 0

        while True:
            x, y, vx, vy = current_state
            action = self.policy[x][y][vx][vy]
            if current_loc in Racetrack.terminal_coord_list:
                break
            elif action == None:
                self.episode.append((current_state, [0, 0]))
                break
            next_state = get_next_state(Racetrack, current_state, action)
            current_loc = (next_state[0], next_state[1])
            self.steps += 1
            self.episode.append((current_state, action))
            current_state = next_state
            # print(f"No crash at step {self.steps} at {current_state} with {action}")
            if self.steps > max_steps:  # to prevent infinite loop in case of bad policy
                logger.debug(
                    "Episode generation stopped after 10000000 steps to prevent infinite loop."
                )
                logger.debug("Last state: %d, %d, %d, %d", *current_state)
                return False
        logger.info("Episode generated successfully in %d steps.", self.steps)
        return True

    def __str__(self):
        return f"Episode(steps={self.steps})"
   
def build_track(og_racetrack):
    racetrack = np.array([list(row) for row in og_racetrack])
    track = np.ones(shape=(len(racetrack), len(racetrack[0])))
    for row in range(len(racetrack)):
        for column in range(len(racetrack[row])):
            if racetrack[row][column] == "#":
                track[row][column] = 0
            elif racetrack[row][column] == "E":
                track[row][column] = 0.4
            elif racetrack[row][column] == "S":
                track[row][column] = 0.6

    return track

def generate_routes_gif(Racetrack, race_track):
    episode = Episode(Racetrack, Racetrack.target_policy_dict)
    track = build_track(race_track)
    gifs_dir = Path(__file__).resolve().parent / "raccetrack_gifs"
    shutil.rmtree(gifs_dir, ignore_errors=True)
    gifs_dir.mkdir(exist_ok=True)
    for each_start in range(len(Racetrack.start_coord_list)):
        images = []
        episode.episode = []
        episode.generate(Racetrack, start_pos=Racetrack.start_coord_list[each_start])
        for step in range(len(episode.episode)):
            track[len(Racetrack.racetrack[0]) - 1 - episode.episode[step][0][1]][
                episode.episode[step][0][0]
            ] = 0.2
            plt.figure(figsize=(10, 10))
            plt.imshow(track)
            plt.title(
                f"Racetrack with start location {Racetrack.start_coord_list[each_start]}",
                fontsize=10,
            )
            output_dir = Path(gifs_dir / f"racetrack_{each_start}")
            output_dir.mkdir(exist_ok=True)

            plt.savefig(output_dir / f"Start-{each_start}-Step-{step}.png")
            # plt.savefig(
            #    f"racetrack_gifs/racetrack_{each_start}/Start-{each_start}-Step-{step}.png"
            # )
            image = Image.open(output_dir / f"Start-{each_start}-Step-{step}.png")
            # image = Image.open(
            #    f"racetrack_gifs/racetrack_{each_start}/Start-{each_start}-Step-{step}.png"
            # )
            images.append(image)
            if (
                race_track[
                    len(Racetrack.racetrack[0]) - 1 - episode.episode[step][0][1]
                ][episode.episode[step][0][0]]
                == "S"
            ):
                track[len(Racetrack.racetrack[0]) - 1 - episode.episode[step][0][1]][
                    episode.episode[step][0][0]
                ] = 0.6
            else:
                track[len(Racetrack.racetrack[0]) - 1 - episode.episode[step][0][1]][
                    episode.episode[step][0][0]
                ] = 1
        images[0].save(
            output_dir
            / f"Optimal_path_for_{Racetrack.start_coord_list[each_start]}.gif",
            save_all=True,
            append_images=images[1:],
            duration=200,
            loop=0,
        )
