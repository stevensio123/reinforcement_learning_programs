import numpy as np
import random
import logging

logger = logging.getlogger(__name__)
logger.setlevel(logging.DEBUG)


def set_file_handler(log_file, level):
    """
    create a file handler for specific levels under one format.
    """
    handler = logging.FileHandler(log_file, mode="w")  # replaces the file on each run
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s - %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    return handler


class Racetrack:
    def __init__(self, racetrack):
        """
        class to represent state space of racetrack.
        state value is initialized to random integer between -5 and 1 (inclusive) for each state.
        """
        logger.info("creating racetrack with input: {}".format(racetrack))
        # reverse and transpose racetrack to match coordinate system (x rows, y rows)
        racetrack_reverse = racetrack[::-1]
        self.racetrack = np.array([list(row) for row in racetrack_reverse]).T

        self.state_values = np.random.randint(
            -5, 0, size=(len(racetrack[0]), len(racetrack), 5, 5)
        )

        self.start_coord_list = []
        self.terminal_coord_list = []
        for i in range(len(self.racetrack)):
            for j in range(len(self.racetrack[i])):
                if self.racetrack[i][j] == "E":
                    self.terminal_coord_list.append((i, j))
                if self.racetrack[i][j] == "S":
                    self.start_coord_list.append((i, j))

        self.x_terminal_loc = self.terminal_coord_list[0][0]
        self.y_terminal_locs = []
        for coord in self.terminal_coord_list:
            self.y_terminal_locs.append(coord[1])
        self.y_terminal_smallest_loc = min(self.y_terminal_locs)

        # create target_policy actions that is greedy to best action
        self.target_policy_dict = np.empty(self.state_values.shape, dtype=object)

    def get_state_value(self, state):
        logger.debug("getting state value of state: {}".format(state))
        x, y, vx, vy = state
        return self.state_values[x, y, vx, vy]
        # equivalent to:
        # return self.state_values[x][y][vx][vy]


def get_next_state(Racetrack, state, a):
    logger.debug("getting next state of {}".format(state))
    x, y, vx, vy = state
    vx += a[0]
    vy += a[1]
    x += vx
    y += vy
    if Racetrack.x_terminal_loc <= x and Racetrack.y_terminal_smallest_loc <= y:
        x = Racetrack.x_terminal_loc
        y = Racetrack.y_terminal_smallest_loc
    try:
        Racetrack.racetrack[x][y]
        logger.debug("next state is {}".format(Racetrack.racetrack[x][y]))
    except IndexError:  # out of bounds
        logger.debug("next state is out of bounds.")
        new_coord = Racetrack.start_coord_list[
            np.random.randint(len(Racetrack.start_coord_list))
        ]
        return (new_coord[0], new_coord[1], 0, 0)
    if Racetrack.racetrack[x][y] == "#":  # crash
        new_coord = Racetrack.start_coord_list[
            np.random.randint(len(Racetrack.start_coord_list))
        ]
        return (new_coord[0], new_coord[1], 0, 0)
    else:
        return (x, y, vx, vy)


def get_action_space(state, state_symbol="N"):
    """
    Takes in state and returns list of possible actions (acceleration) that can be taken from that state.
    Acceleration can be -1, 0, or 1 in both x and y, and velocity < 5
    """
    action_space = []
    accel = [-1, 0, 1]
    x, y, vx, vy = state
    for horizontal in accel:
        if 0 <= (vx + horizontal) < 5:
            for vertical in accel:
                if 0 < (vy + vertical) < 5 or (
                    (state_symbol == "S" or (vx + horizontal) != 0)
                    and 0 <= (vy + vertical) < 5
                ):
                    action_space.append([horizontal, vertical])
    return action_space


def get_policy(obj: Racetrack, epsilon=0.1):
    """
    takes in a StateSpace object and epsilon value,
    returns an array with the same shape as the state space,
    with the chosen action for each state according to the behavior policy as the elements.
    """
    state_values = obj.state_values

    # use object array to store lists as elements (because there are two actions)
    policy = np.empty(state_values.shape, dtype=object)
    for x in range(state_values.shape[0]):
        for y in range(state_values.shape[1]):
            for vx in range(state_values.shape[2]):
                for vy in range(state_values.shape[3]):
                    state = (x, y, vx, vy)
                    # print(state)
                    # choose optimal / random action:
                    # print(f"{obj.racetrack[x][y]}")
                    action_space_ls = get_action_space(state, obj.racetrack[x][y])
                    if random.random() > epsilon:
                        # take optimal action according to current state values
                        action_idx = get_optimal_action(obj, state, action_space_ls)
                        policy[x][y][vx][vy] = action_space_ls[action_idx]
                    else:
                        # take random action
                        action_idx = np.random.randint(len(action_space_ls))
                        policy[x][y][vx][vy] = action_space_ls[action_idx]
    return policy


def get_optimal_action(Racetrack, state, action_space_ls):
    """
    takes in a StateSpace object and epsilon value,
    returns an array with the same shape as the state space,
    with the chosen action for each state according to the behavior policy as the elements.
    """
    state_values = Racetrack.state_values
    action_values = []
    for action in action_space_ls:
        next_state_idx = get_next_state(Racetrack, state, action)
        next_state_value = state_values[next_state_idx]
        action_values.append(next_state_value)
    action_idx = np.where(action_values == np.max(action_values))[0][-1]
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

    def generate(self, Racetrack, max_steps=100000):
        """
        method to create episode by following the policy until it reaches terminal state.
        """
        current_loc = Racetrack.start_coord_list[
            np.random.randint(len(Racetrack.start_coord_list))
        ]
        current_state = (current_loc[0], current_loc[1], 0, 0)
        self.steps = 0

        while True:
            x, y, vx, vy = current_state
            action = self.policy[x][y][vx][vy]
            next_state = get_next_state(Racetrack, current_state, action)
            if current_loc in Racetrack.terminal_coord_list:
                break
            current_loc = (next_state[0], next_state[1])
            self.steps += 1
            self.episode.append((current_state, action))
            current_state = next_state
            # print(f"No crash at step {self.steps} at {current_state} with {action}")
            if self.steps > max_steps:  # to prevent infinite loop in case of bad policy
                print(
                    "Episode generation stopped after 10000000 steps to prevent infinite loop."
                )
                print(f"    Last state: {current_state}")
                return False
        print("Episode generated")
        print(f"    Steps taken: {self.steps}")
        return True

    def __str__(self):
        print(f"Episode steps: {self.steps}")
        print(
            f"Episode trajectory (first 3 steps): {self.episode[:3]}"
        )  # print first 3 steps of episode
        print(
            f"Episode trajectory (last 3 steps): {self.episode[-3:]}"
        )  # print last 3 steps of episode
        return f"Total steps generated: {self.steps}"
