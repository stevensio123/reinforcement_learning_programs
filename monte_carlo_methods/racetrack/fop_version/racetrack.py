import numpy as np
import pandas as pd
import random

# initialize variables
<<<<<<< HEAD
race_track = ["########",
              "#NNNNNNE",
              "#NN#####",
              "#NN#####",
              "#SS#####"]
=======
race_track = ["########", "#NNNNNNE", "#NN#####", "#NN#####", "#SS#####"]
>>>>>>> 7d7335ae02547e93b4b33ea2faabd4a2c1d68998
epsilon = 0.2
gamma = 0.9
# To find starting points
start_coord_list = []
end_coord_list = []
racetrack_list = []
# to convert race_track into coordinate system
# i represent y-axis coordinate, j represent x-axis
for i in race_track:
    racetrack_list.append(list(i))

# Convert racetrack_list into a numpy array for transpose and reverse
race_track_array = np.array(racetrack_list)
race_track_array = np.flip(race_track_array, axis=0).transpose()
<<<<<<< HEAD
racetrack_list = race_track_array.tolist() # Converts it back to list for our ref
=======
racetrack_list = race_track_array.tolist()  # Converts it back to list for our ref
>>>>>>> 7d7335ae02547e93b4b33ea2faabd4a2c1d68998

# to find start_coord_list & end_coord_list
for i in range(len(racetrack_list)):
    for j in range(len(racetrack_list[i])):
<<<<<<< HEAD
        if racetrack_list[i][j] == 'S':
            start_coord_list.append([i,j])
        if racetrack_list[i][j] == 'E':
            end_coord_list.append([i,j])

'''
# State_space is state_value
# dimensions represent:
rows of race track, columns of race track, row-acceleration, col-acceleration
'''
state_space = np.random.randint(1,3,(len(racetrack_list),len(racetrack_list[0]),5,5))
# cumulative sum sampling ratio: state = 4d, action = 3d; total 7d
cum_IS = np.zeros((len(racetrack_list),len(racetrack_list[0]),5,5,3,3),int)

print(f"state space shape (x rows, y rows, x velocity, y velocity): {state_space.shape}")
print(f"cumulative importance sampling shape (x rows, y rows, x velocity, y velocity, x acceleration, y acceleration): {cum_IS.shape}")
=======
        if racetrack_list[i][j] == "S":
            start_coord_list.append([i, j])
        if racetrack_list[i][j] == "E":
            end_coord_list.append([i, j])

"""
# State_space is state_value
# dimensions represent:
rows of race track, columns of race track, row-acceleration, col-acceleration
"""
state_space = np.random.randint(
    1, 3, (len(racetrack_list), len(racetrack_list[0]), 5, 5)
)
# cumulative sum sampling ratio: state = 4d, action = 3d; total 7d
cum_IS = np.zeros((len(racetrack_list), len(racetrack_list[0]), 5, 5, 3, 3), int)

print(
    f"state space shape (x rows, y rows, x velocity, y velocity): {state_space.shape}"
)
print(
    f"cumulative importance sampling shape (x rows, y rows, x velocity, y velocity, x acceleration, y acceleration): {cum_IS.shape}"
)
>>>>>>> 7d7335ae02547e93b4b33ea2faabd4a2c1d68998
print(f"start coordinates: {start_coord_list}")
print(f"end coordinates: {end_coord_list}")


def action_space(state):
<<<<<<< HEAD
  action_space = []
  accel = [-1,0,1]
  x, y, vx, vy = state
  for horizontal in accel:
    if 0 <= (vx + horizontal) < 5:
      for vertical in accel:
        if 0 <= (vy + vertical) < 5:
          action_space.append([horizontal,vertical])
  return action_space

def next_state(state, a):
  x, y, vx, vy = state
  vx += a[0]
  vy += a[1]
  x += vx
  y += vy
  try:
    racetrack_list[x][y]
  except IndexError:
    start_coord = start_coord_list[np.random.randint(len(start_coord_list))] # randomise starting position
    return (start_coord[0],start_coord[1],0,0)
  if racetrack_list[x][y] == '#':
    start_coord = start_coord_list[np.random.randint(len(start_coord_list))] # randomise starting position
    return (start_coord[0],start_coord[1],0,0)
  else:
    return (x,y,vx,vy)

def behavior_policy(state):
  # choose optimal / random action:
  action_space_ls = action_space(state)
  #optimal_prob = (1 - epsilon) + (epsilon / len(action_space_ls))
  #random_prob =  (epsilon / len(action_space_ls))
  if random.random() > epsilon:
    state_value_ls = []
    for action in action_space_ls:
      next_state_idx = next_state(state, action)
      #print(next_state_idx)
      state_value_ls.append(state_space[next_state_idx])
    # to break ties by taking most progressive action
    action_idx = np.where(state_value_ls == np.max(state_value_ls))[0][-1]
    return(action_space_ls[action_idx])
  else:
    action_idx = np.random.randint(len(action_space_ls))
    return(action_space_ls[action_idx])

def generate_episode(policy):
  # no need to generate Reward (all = -1)
  # TASK: think of using what data type to store this output
  start_coord = start_coord_list[np.random.randint(len(start_coord_list))] # randomise starting position
  current_state = (start_coord[0],start_coord[1],0,0)
  current_coordinate = [start_coord[0],start_coord[1]]
  episode = []
  counter = 0
  while current_coordinate not in end_coord_list:
    action = behavior_policy(current_state)
    episode.append((current_state,action))
    new_state = next_state(current_state,action)
    current_state = new_state
    current_coordinate = [current_state[0],current_state[1]]
    counter += 1
 # [[s,a],...]
  episode.append((current_state,[0,0]))
  print(f"episode generated, ({counter} time steps)")
  return episode

#episode = generate_episode(behavior_policy)

def incremental_prediction(episode):
  episode.reverse()
  G = 0
  W = 1
  for step in range(len(episode)):
    G = (gamma*G) - 1
    y,x,v1,v2 = episode[step][0] # state
    a1,a2 = episode[step][1] # action
    cum_IS[y,x,v1,v2,a1,a2] += (cum_IS[y,x,v1,v2,a1,a2] + W)
=======
    action_space = []
    accel = [-1, 0, 1]
    x, y, vx, vy = state
    for horizontal in accel:
        if 0 <= (vx + horizontal) < 5:
            for vertical in accel:
                if 0 <= (vy + vertical) < 5:
                    action_space.append([horizontal, vertical])
    return action_space


def next_state(state, a):
    x, y, vx, vy = state
    vx += a[0]
    vy += a[1]
    x += vx
    y += vy
    try:
        racetrack_list[x][y]
    except IndexError:
        start_coord = start_coord_list[
            np.random.randint(len(start_coord_list))
        ]  # randomise starting position
        return (start_coord[0], start_coord[1], 0, 0)
    if racetrack_list[x][y] == "#":
        start_coord = start_coord_list[
            np.random.randint(len(start_coord_list))
        ]  # randomise starting position
        return (start_coord[0], start_coord[1], 0, 0)
    else:
        return (x, y, vx, vy)


def behavior_policy(state):
    # choose optimal / random action:
    action_space_ls = action_space(state)
    # optimal_prob = (1 - epsilon) + (epsilon / len(action_space_ls))
    # random_prob =  (epsilon / len(action_space_ls))
    if random.random() > epsilon:
        state_value_ls = []
        for action in action_space_ls:
            next_state_idx = next_state(state, action)
            # print(next_state_idx)
            state_value_ls.append(state_space[next_state_idx])
        # to break ties by taking most progressive action
        action_idx = np.where(state_value_ls == np.max(state_value_ls))[0][-1]
        return action_space_ls[action_idx]
    else:
        action_idx = np.random.randint(len(action_space_ls))
        return action_space_ls[action_idx]


def generate_episode(policy):
    # no need to generate Reward (all = -1)
    # TASK: think of using what data type to store this output
    start_coord = start_coord_list[
        np.random.randint(len(start_coord_list))
    ]  # randomise starting position
    current_state = (start_coord[0], start_coord[1], 0, 0)
    current_coordinate = [start_coord[0], start_coord[1]]
    episode = []
    counter = 0
    while current_coordinate not in end_coord_list:
        action = behavior_policy(current_state)
        episode.append((current_state, action))
        new_state = next_state(current_state, action)
        current_state = new_state
        current_coordinate = [current_state[0], current_state[1]]
        counter += 1
    # [[s,a],...]
    episode.append((current_state, [0, 0]))
    print(f"episode generated, ({counter} time steps)")
    return episode


# episode = generate_episode(behavior_policy)


def incremental_prediction(episode):
    episode.reverse()
    G = 0
    W = 1
    for step in range(len(episode)):
        G = (gamma * G) - 1
        y, x, v1, v2 = episode[step][0]  # state
        a1, a2 = episode[step][1]  # action
        cum_IS[y, x, v1, v2, a1, a2] += cum_IS[y, x, v1, v2, a1, a2] + W

>>>>>>> 7d7335ae02547e93b4b33ea2faabd4a2c1d68998

incremental_prediction(episode)
print(episode[-1])

<<<<<<< HEAD
def q(s,a):
  # return current estimate of the value of the next state
  return state_space[next_state(s,a)]

def incremental_prediction(episode):
  episode.reverse()
  G = 0
  W = 1
  for step in range(len(episode)):
    G = (gamma*G) - 1
    y,x,v1,v2 = episode[step][0] # state
    a1,a2 = episode[step][1] # action
    cum_IS[y,x,v1,v2,a1,a2] += (cum_IS[y,x,v1,v2,a1,a2] + W)
=======

def q(s, a):
    # return current estimate of the value of the next state
    return state_space[next_state(s, a)]


def incremental_prediction(episode):
    episode.reverse()
    G = 0
    W = 1
    for step in range(len(episode)):
        G = (gamma * G) - 1
        y, x, v1, v2 = episode[step][0]  # state
        a1, a2 = episode[step][1]  # action
        cum_IS[y, x, v1, v2, a1, a2] += cum_IS[y, x, v1, v2, a1, a2] + W

>>>>>>> 7d7335ae02547e93b4b33ea2faabd4a2c1d68998

state = (1, 1, 3, 2)
a = (-1, 1)


def main():

<<<<<<< HEAD
  print(state_space[next_state(state, a)])


main()
=======
    print(state_space[next_state(state, a)])


main()

>>>>>>> 7d7335ae02547e93b4b33ea2faabd4a2c1d68998
