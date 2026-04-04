import numpy as np
import random

def get_next_state(racetrack, state, a):
    x, y, vx, vy = state
    vx += a[0]
    vy += a[1]
    x += vx
    y += vy
    try:
        racetrack[x][y]
    except IndexError: # out of bounds
        return False 
    if racetrack[x][y] == '#': # crash
        return False
    else:
        return x,y,vx,vy

def get_action_space(state):
    """
    Takes in state and returns list of possible actions (acceleration) that can be taken from that state.
    Acceleration can be -1, 0, or 1 in both x and y, and velocity < 5
    """
    action_space = []
    accel = [-1,0,1]
    x, y, vx, vy = state
    for horizontal in accel:
        if 0 <= (vx + horizontal) < 5:
            for vertical in accel:
                if 0 <= (vy + vertical) < 5:
                    action_space.append([horizontal,vertical])
    return action_space

def behavior_policy(obj: StateSpace, epsilon=0.1):
    """
    takes in a StateSpace object and epsilon value, 
    returns an array with the same shape as the state space,
    with the chosen action for each state according to the behavior policy as the elements.
    """
    state_values = obj.state_values
    racetrack = obj.racetrack
    start_locs = obj.start_locs()
    #print(start_locs)
    # use object array to store lists as elements (because there are two actions)
    policy = np.empty(state_values.shape, dtype=object) 
    for x in range(state_values.shape[0]):
        for y in range(state_values.shape[1]):
            for vx in range(state_values.shape[2]):
                for vy in range(state_values.shape[3]):
                    state = (x,y,vx,vy)
                    # choose optimal / random action:
                    action_space_ls = get_action_space(state)
                    if random.random() > epsilon:
                        # take optimal action according to current state values
                        action_values = []
                        for action in action_space_ls:
                            next_state_idx = get_next_state(racetrack, state, action)
                            if next_state_idx == False:
                                # get value of random starting state if crash or out of bounds
                                start_loc = start_locs[np.random.randint(len(start_locs))]
                                next_state_idx = (start_loc[0], start_loc[1], 0, 0)   

                                next_state_value = state_values[next_state_idx]
                            else:
                                # print(next_state_idx)
                                next_state_value = state_values[next_state_idx]
                            action_values.append(next_state_value)
                        # to break ties by taking most progressive action
                        action_idx = np.where(action_values == np.max(action_values))[0][-1]
                        policy[x][y][vx][vy] = action_space_ls[action_idx]
                    else:
                        # take random action
                        action_idx = np.random.randint(len(action_space_ls))
                        policy[x][y][vx][vy] = action_space_ls[action_idx]
    return policy


class Racetrack():
    def __init__(self, racetrack):
        # reverse and transpose racetrack to match coordinate system (x rows, y rows)
        racetrack_reverse = racetrack[::-1]
        self.racetrack = np.array([list(row) for row in racetrack_reverse]).T

    def start_locs(self):
        start_coord_list = []
        for i in range(len(self.racetrack)):
            for j in range(len(self.racetrack[i])):
                if self.racetrack[i][j] == 'S':
                    start_coord_list.append([i,j])
        return start_coord_list
    def terminal_locs(self):
        end_coord_list = []
        for i in range(len(self.racetrack)):
            for j in range(len(self.racetrack[i])):
                if self.racetrack[i][j] == 'E':
                    end_coord_list.append([i,j])
        return end_coord_list

class StateSpace(Racetrack):
    """
    class to represent state space of racetrack.
    state value is initialized to random integer between -5 and 1 (inclusive) for each state.
    """
    def __init__(self, racetrack):
        # dimensions represent:
        # rows of race track, columns of race track, row-acceleration, col-acceleration
        super().__init__(racetrack)
        self.state_values = np.random.randint(-5,2,size=(len(racetrack[0]),len(racetrack),5,5))

    def get_state_value(self, state):
        x, y, vx, vy = state
        return self.state_values[x][y][vx][vy]


class Episode(Racetrack):
    """
    inherits from Racetrack class, creates episode by following the policy until it reaches terminal state.
    takes in racetrack and policy as arguments.
    episode is stored as a list of (state, action) pairs.
    """
    def __init__(self, racetrack, policy):
        """
        start_locs: list of starting coordinates
        terminal_locs: list of terminal coordinates
        policy: function that takes in state and outputs action
        """
        super().__init__(racetrack)
        self.start_loc = self.start_locs()[np.random.randint(len(self.start_locs()))]
        self.episode = []
        self.policy = policy
        self.steps = 0

    def generate(self):
        """
        method to create episode by following the policy until it reaches terminal state.
        """
        terminal_locs = self.terminal_locs()
        current_loc = self.start_loc
        current_state = (self.start_loc[0],self.start_loc[1],0,0)
        while current_loc not in terminal_locs:
            action = self.policy(current_state)
            self.episode.append((current_state,action))
            current_state = get_next_state(self.racetrack, self.current_state, action)
            if current_state == False: 
            # if crash or out of bounds, reset to random start
                current_state = self.start_locs()[np.random.randint(len(self.start_locs()))]
            else:
                current_state = get_next_state(self.racetrack, self.current_state, action)
            current_loc = [current_state[0],current_state[1]]
            self.steps += 1
    def __str__(self):
        return f"Total steps generated: {self.steps}"
