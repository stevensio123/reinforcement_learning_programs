import numpy as np
import racetrack_utils as utils

def incremental_prediction(Racetrack, episode, cum_IS, epsilon, gamma = 0.9):
  
  episode.reverse()
  G = 0
  W = 1
  for step in range(len(episode) - 1):
    # G already accounts for reward -1 for each step
    step += 1
    G = (gamma*G) - 1
    x, y, v1, v2 = episode[step][0] # state
    a1,a2 = episode[step][1] # action
    cum_IS[x,y,v1,v2,a1,a2] += W

    print(f"step {step} coordinate: ({x}, {y}, {v1}, {v2}) with action: ({a1}, {a2})")

    # update q(s,a)
    next_state = utils.get_next_state(Racetrack, episode[step][0], episode[step][1])
    Racetrack.state_values[x][y][v1][v2] = Racetrack.get_state_value(next_state) + ((W / (cum_IS[x,y,v1,v2,a1,a2])) * (G - Racetrack.get_state_value(next_state)))

    # update target policy action based on best value of actions
    action_space_ls = utils.get_action_space(episode[step][0], Racetrack.racetrack[x][y])
    # take optimal action according to current state values
    optimal_action_idx = utils.get_optimal_action(Racetrack, episode[step][0], action_space_ls)
    Racetrack.target_policy_dict[x][y][v1][v2] = action_space_ls[optimal_action_idx]

    print(f"Action state value: {Racetrack.get_state_value(next_state)}")
    print(f"Target policy action: {Racetrack.target_policy_dict[x][y][v1][v2]} with state value: {Racetrack.get_state_value(utils.get_next_state(Racetrack, episode[step][0], Racetrack.target_policy_dict[x][y][v1][v2]))}")

    # compare if target_policy action matches current action taken
    if Racetrack.target_policy_dict[x][y][v1][v2] != episode[step][1]:
      return False
    
    W = W / ((1- epsilon) + (epsilon / len(action_space_ls)))


def off_policy_control(Racetrack, epsilon = 0.1, max_episode_count = 1000, max_episode_generation_attempt = 10):
  # cumulative sum sampling ratio: state = 4d, action = 3d; total 7d
  cum_IS = np.zeros((len(Racetrack.racetrack),len(Racetrack.racetrack[0]),5,5,3,3),int)
  behaviour_policy = utils.get_policy(Racetrack, epsilon)
  episode_count = 0
  ep_generation_attempt = 0
  episode = utils.Episode(Racetrack, behaviour_policy)
  while True:
    episode.policy = behaviour_policy
    episode.episode = []
    if episode.generate(Racetrack):
      episode_count += 1
      if incremental_prediction(Racetrack, episode.episode, cum_IS, epsilon) == False:
        behaviour_policy = utils.get_policy(Racetrack, epsilon)
        continue
      if episode_count >= max_episode_count:
        print("Off-policy control failed, retrying with new episode generation...")
        continue
      print(f"Off-policy control successful with episode count: {episode_count}")
      break
    else:
      ep_generation_attempt += 1
      print(f"Episode generation reached max step, generation attempt {ep_generation_attempt} failed...")
      if ep_generation_attempt > max_episode_generation_attempt:
        if epsilon < 1:
          epsilon += 0.01
          print(f"Retrying with new behaviour policy with epsilon value of {round(epsilon,2)} ....")
        else:
          print("Epsilon value at maximum of 0.99, stopping off-policy control.")
          break
        ep_generation_attempt = 0
      behaviour_policy = utils.get_policy(Racetrack, epsilon)
  
  print("off-policy completed.")




def main():
  """"
  race_track = ["########",
                "#NNNNNNE",
                "#NNNNNNE",
                "#NN#####",
                "#SS#####"]
  """

  race_track = ["#######E",
                "#NNNNNNE",
                "#NNNNNNE",
                "#NNNNNNE",
                "#SS#####"]

  race_track_obj = utils.Racetrack(race_track)

  off_policy_control(race_track_obj, max_episode_count=10, max_episode_generation_attempt=4)
  

main()