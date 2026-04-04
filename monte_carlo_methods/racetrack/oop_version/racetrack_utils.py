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
        while current_loc not in terminal_locs
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
