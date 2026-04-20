import numpy as np
import racetrack_utils as utils

def incremental_prediction(Racetrack, episode, cum_IS, gamma = 0.9):
  
  episode.reverse()
  G = 0
  W = 1
  next_state = utils.get_next_state(Racetrack.racetrack, episode[-1][0], episode[-1][1])
  for step in range(len(episode) + 1):
    # G already accounts for reward -1 for each step
    G = (gamma*G) - 1
    x, y, v1, v2 = episode[step][0] # state
    a1,a2 = episode[step][1] # action
    cum_IS[x,y,v1,v2,a1,a2] += W

    # update q(s,a)
    Racetrack.state_values[x][y][v1][v2] = Racetrack.get_state_value(next_state) + ((W / (cum_IS[x,y,v1,v2,a1,a2])) * (G - Racetrack.get_state_value(next_state)))

    # update target policy action based on best value of actions
    action_space_ls = utils.get_action_space(episode[step][0], Racetrack.racetrack[x][y])
    # take optimal action according to current state values
    action_values = utils.get_action_values(action_space_ls, Racetrack.racetrack, episode[step][0], Racetrack.start_coord_list, Racetrack.state_values)
    # to break ties by taking most progressive action, np.where returns an array
    action_idx = np.where(action_values == np.max(action_values))[0][-1]
    Racetrack.target_policy_dict[x][y][v1][v2] = action_space_ls[action_idx]
    print(Racetrack.target_policy_dict[x][y][v1][v2])

    # compare if target_policy action matches current action taken
    if Racetrack.target_policy_dict[x][y][v1][v2] != episode[step][1]:
      return False

    # for next state
    next_state = episode[step][0]


def policy_control(Racetrack, behaviour_policy):
  # cumulative sum sampling ratio: state = 4d, action = 3d; total 7d
  cum_IS = np.zeros((len(Racetrack.racetrack),len(Racetrack.racetrack[0]),5,5,3,3),int)
  episode_count = 0
  while True:
    episode_count += 1
    print(f"current episode count: {episode_count}")
    episode = utils.Episode(Racetrack, behaviour_policy)
    if episode.generate(Racetrack):
      incremental_prediction(Racetrack, episode.episode, cum_IS)


def main():
  """"
  race_track = ["########",
                "#NNNNNNE",
                "#NNNNNNE",
                "#NN#####",
                "#SS#####"]
  """

  race_track = ["####EEEE",
                "#NNNNNNE",
                "#NNNNNNE",
                "#NNNNNNE",
                "#NNNNN##",
                "#SSSSS##"]

  race_track_obj = utils.Racetrack(race_track)

  target_policy = utils.target_policy(race_track_obj)

  behavior_policy = utils.behavior_policy(race_track_obj)

  policy_control(race_track_obj, behavior_policy)
  

main()