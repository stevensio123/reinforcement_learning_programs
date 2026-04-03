import numpy as np
class racetrack():
    def __init__(self, racetrack):
        self.racetrack = np.array([list(row) for row in racetrack]).T

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


def next_state(racetrack, state, a):
    x, y, vx, vy = state
    vx += a[0]
    vy += a[1]
    x += vx
    y += vy
    try:
        return racetrack[x][y]
    except IndexError: # out of bounds
        return False 
    if racetrack[x][y] == '#': # crash
        return False
    else:
        return (x,y,vx,vy)
"""
# Done to aid in each instances of a repeated function
class episode():
  # Use to just keep track of all episodes created throughout the algorithm
  total_episodes = 0

  # Creates the instances with starting attributes
  def __init__(self, start_locs, terminal_locs, policy):
  # no need to generate Reward (all = -1)
    self.start_loc = start_locs[np.random.randint(len(start_locs))]    
    self.current_state = (self.start_loc[0],self.start_loc[1],0,0)
    self.current_loc = self.start_loc
    self.next_state = ()
    self.episode = []
    self.policy = policy
    self.counter = 0

  # This can be the method to call for generating the episode
  def generate(self):
    while self.current_loc not in end_coord_list:
      action = behavior_policy(self.current_state)
      self.episode.append((self.current_state,action))
      
      self.new_state = next_state(self.current_state,action)
      self.current_state = self.new_state
      self.current_coordinate = [self.current_state[0],self.current_state[1]]
      self.counter += 1
  # [[s,a],...]
    self.episode.append((self.current_state,[0,0]))
    generate_episode.total_episodes += 1
    print(f"episode generated, took ({self.counter} time steps)")

  # To set out tracking statements, we can expand if needed.
  def __str__(self):
    return f"Total episodes generated: {generate_episode.total_episodes}"
"""