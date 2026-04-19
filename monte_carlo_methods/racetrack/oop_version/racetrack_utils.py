import numpy as np
import random

class Racetrack():
    def __init__(self, racetrack):
        """
        class to represent state space of racetrack.
        state value is initialized to random integer between -5 and 1 (inclusive) for each state.
        """
        # reverse and transpose racetrack to match coordinate system (x rows, y rows)
        racetrack_reverse = racetrack[::-1]
        self.racetrack = np.array([list(row) for row in racetrack_reverse]).T

        self.start_coord_list = []
        self.terminal_coord_list = []
        for i in range(len(self.racetrack)):
            for j in range(len(self.racetrack[i])):
                if self.racetrack[i][j] == 'E':
                    self.terminal_coord_list.append([i,j])
                if self.racetrack[i][j] == 'S':
                    self.start_coord_list.append([i,j])
        
        self.state_values = np.random.randint(-5,2,size=(len(racetrack[0]),len(racetrack),5,5))

    def get_state_value(self, state):
        x, y, vx, vy = state
        return self.state_values[x,y,vx,vy]
        # equivalent to:
        # return self.state_values[x][y][vx][vy]


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

def get_optimal_action(state, Racetrack, action_space_ls):
    racetrack = Racetrack.racetrack
    state_values = Racetrack.state_values
    start_locs = Racetrack.start_coord_list
    # choose optimal / random action:
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
    # to break ties by taking most progressive action, np.where returns an array
    action_idx = np.where(action_values == np.max(action_values))[0][-1]
    return action_idx

def get_policy(obj: Racetrack, epsilon=0.1):
    """
    takes in a StateSpace object and epsilon value, 
    returns an array with the same shape as the state space,
    with the chosen action for each state according to the behavior policy as the elements.
    """
    state_values = obj.state_values
    racetrack = obj.racetrack
    start_locs = obj.start_coord_list

    # use object array to store lists as elements (because there are two actions)
    policy = np.empty(state_values.shape, dtype=object) 
    for x in range(state_values.shape[0]):
        for y in range(state_values.shape[1]):
            for vx in range(state_values.shape[2]):
                for vy in range(state_values.shape[3]):
                    action_space_ls = get_action_space((x,y,vx,vy))
                    if np.random.random() < epsilon:
                        # take optimal action
                        action_idx = get_optimal_action((x,y,vx,vy), obj, action_space_ls)
                        policy[x][y][vx][vy] = action_space_ls[action_idx]
                    else:
                        # take random action
                        action_idx = np.random.randint(len(action_space_ls))
                        policy[x][y][vx][vy] = action_space_ls[action_idx]
    return policy

class Episode():
    """
    Racetrack object and policy as arguments, this is so we can access the start/terminal locs method.
    creates episode by following the policy until it reaches terminal state.
    episode is stored as a list of tuples of: (state, action) pairs.
    """
    def __init__(self, Racetrack, policy):
        """
        policy: a 4D array that has an action for each state (x, y, vx, vy)
        super().__init__(racetrack)
        """
        # start_loc: randomly chosen starting coordinate 
        self.start_loc = Racetrack.start_coord_list[np.random.randint(len(Racetrack.start_coord_list))]
        self.terminal_locs = Racetrack.terminal_coord_list # Moved to init as useful
        self.episode = []
        self.policy = policy
        self.steps = 1

    def generate(self, Racetrack, max_steps=100000):
        """
        method to create episode by following the policy until it reaches terminal state.
        """
        current_loc = self.start_loc
        current_state = (self.start_loc[0],self.start_loc[1],0,0)
        while current_loc not in self.terminal_locs:
            x,y,vx,vy = current_state
            action = self.policy[x][y][vx][vy]
            self.episode.append((current_state,action))
            current_state = get_next_state(Racetrack.racetrack, current_state, action)
            if current_state == False: 
            # if crash or out of bounds, reset to random start
                current_loc = Racetrack.start_coord_list[np.random.randint(len(Racetrack.start_coord_list))]
                current_state = (current_loc[0], current_loc[1], 0, 0)
            else:
                current_loc = [current_state[0],current_state[1]]
            self.steps += 1
            if self.steps > max_steps: # to prevent infinite loop in case of bad policy
                print("Episode generation stopped after 10000000 steps to prevent infinite loop.")
                print(f"Last state: {current_state}")
                return False

    def __str__(self):
        print(f"Episode steps: {self.steps}")
        print(f"Episode trajectory (first 3 steps): {self.episode[:3]}") # print first 3 steps of episode
        print(f"Episode trajectory (last 3 steps): {self.episode[-3:]}") # print last 3 steps of episode
        return f"Total steps generated: {self.steps}"

