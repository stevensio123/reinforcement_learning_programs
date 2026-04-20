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
        return (x,y,vx,vy)

def get_action_space(state, state_symbol="N"):
    """
    Takes in state and returns list of possible actions (acceleration) that can be taken from that state.
    Acceleration can be -1, 0, or 1 in both x and y, and velocity < 5
    """
    action_space = []
    accel = [-1,0,1]
    x, y, vx, vy = state
    for horizontal in accel:
        if 0 < (vx + horizontal) < 5 or (state_symbol == "S" and 0 <= (vx + horizontal) < 5):
            for vertical in accel:
                if 0 < (vy + vertical) < 5 or (state_symbol == "S" and 0 <= (vy + vertical) < 5):
                    action_space.append([horizontal,vertical])
    return action_space

def get_action_values(action_space_ls, racetrack, state, start_locs, state_values):
    action_values = []
    for action in action_space_ls:
        next_state_idx = get_next_state(racetrack, state, action)
        if next_state_idx == False:
            # get value of random starting state if crash or out of bounds
            start_loc = start_locs[np.random.randint(len(start_locs))]
            next_state_idx = (start_loc[0], start_loc[1], 0, 0)   
            next_state_value = state_values[next_state_idx]
        else:
            next_state_value = state_values[next_state_idx]
        action_values.append(next_state_value)
    return action_values

class Racetrack():
    def __init__(self, racetrack):
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
        """
        class to represent state space of racetrack.
        state value is initialized to random integer between -5 and 1 (inclusive) for each state.
        NEW, combined Racetrack and State_space class as these only need to be called once
        """
        self.state_values = np.random.randint(-5,2,size=(len(racetrack[0]),len(racetrack),5,5))

        # create target_policy actions that is greedy to best action
        self.target_policy_dict = np.empty(self.state_values.shape, dtype=object)

    def get_state_value(self, state):
        x, y, vx, vy = state
        return self.state_values[x][y][vx][vy]


def get_policy(obj: Racetrack, epsilon=0.1):
    """
    takes in a StateSpace object and epsilon value, 
    returns an array with the same shape as the state space,
    with the chosen action for each state according to the behavior policy as the elements.
    """
    state_values = obj.state_values
    racetrack = obj.racetrack
    start_locs = obj.start_coord_list

    #print(start_locs)
    # use object array to store lists as elements (because there are two actions)
    policy = np.empty(state_values.shape, dtype=object) 
    for x in range(state_values.shape[0]):
        for y in range(state_values.shape[1]):
            for vx in range(state_values.shape[2]):
                for vy in range(state_values.shape[3]):
                    state = (x,y,vx,vy)
                    #print(state)
                    # choose optimal / random action:
                    #print(f"{obj.racetrack[x][y]}")
                    action_space_ls = get_action_space(state, obj.racetrack[x][y])
                    if random.random() > epsilon:
                        # take optimal action according to current state values
                        action_idx = get_optimal_action(Racetrack, state, action_space_ls)
                        policy[x][y][vx][vy] = action_space_ls[action_idx]
                    else:
                        # take random action
                        action_idx = np.random.randint(len(action_space_ls))
                        policy[x][y][vx][vy] = action_space_ls[action_idx]
    print(f"policy at [1][0][0][0]: {policy[1][0][0][0]}")
    return policy

def get_optimal_action(obj: Racetrack, state, action_space_ls):
    """
    takes in a StateSpace object and epsilon value, 
    returns an array with the same shape as the state space,
    with the chosen action for each state according to the behavior policy as the elements.
    """
    state_values = obj.state_values

    x, y, vx, vy = state
    action_values = get_action_values(action_space_ls, Racetrack.racetrack, state, Racetrack.start_locs, state_values)
    action_idx = np.where(action_values == np.max(action_values))[0][-1]
    print(f"policy at [1][0][0][0]: {Racetrack.target_policy_dict[1][0][0][0]}")
    return action_idx
    

class Episode():
    """
    takes the Racetrack class as an input parameter, this is so we can access the start/terminal locs method.
    creates episode by following the policy until it reaches terminal state.
    takes in racetrack and policy as arguments.
    episode is stored as a list of (state, action) pairs.
    """
    def __init__(self, racetrack, policy):
        """
        policy: a 4D array that has an action for each state (x, y, vx, vy)
        super().__init__(racetrack)
        """
        # start_loc: randomly chosen starting coordinate 
        self.start_loc = racetrack.start_coord_list[np.random.randint(len(racetrack.start_coord_list))]
        self.terminal_locs = racetrack.terminal_coord_list # Moved to init as useful
        self.episode = []
        self.policy = policy
        self.steps = 1

    def generate(self, Racetrack):
        """
        method to create episode by following the policy until it reaches terminal state.
        """
        current_loc = self.start_loc
        current_state = (self.start_loc[0],self.start_loc[1],0,0)
        while current_loc not in self.terminal_locs:
            x, y, vx, vy = current_state
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
            if self.steps > 10000000: # to prevent infinite loop in case of bad policy
                print("Episode generation stopped after 10000000 steps to prevent infinite loop.")
                print(f"Last state: {current_state}")
                print("New behavious policy generated")
                self.policy = get_policy(Racetrack)
                return False
        print("Episode generated")
        print(f"Steps taken: {self.steps}")
        return True

    def __str__(self):
        print(f"Episode steps: {self.steps}")
        print(f"Episode trajectory (first 10 steps): {self.episode[:10]}") # print first 10 steps of episode
        print(f"Episode trajectory (last 3 steps): {self.episode[-3:]}") # print last 3 steps of episode
        return f"Total steps generated: {self.steps}"
