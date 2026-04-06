import numpy as np
import racetrack_utils as utils

gamma = 0.9

def incremental_prediction(Racetrack, episode, cum_IS):
  
  episode.reverse()
  G = 0
  W = 1
  for step in range(len(episode)):
    G = (gamma*G) - 1
    y,x,v1,v2 = episode[step][0] # state
    a1,a2 = episode[step][1] # action
    cum_IS[y,x,v1,v2,a1,a2] += (cum_IS[y,x,v1,v2,a1,a2] + W)
    # find q(s,a) of 
    next_state = utils.get_next_state(Racetrack.racetrack, s, a)
    state_value = Racetrack.get_state_value(next_state)


def policy_control(Racetrack, behaviour_policy):
  # cumulative sum sampling ratio: state = 4d, action = 3d; total 7d
  cum_IS = np.zeros((len(Racetrack.racetrack),len(Racetrack.racetrack[0]),5,5,3,3),int)
  while True:
    episode_count = 0
    episode = utils.Episode(Racetrack, behaviour_policy)
    if episode.generate(Racetrack):
      episode_count += 1
      incremental_prediction(episode, cum_IS)
    else:
      pass

state = (1, 1, 1, 1)
a = (-1, 1)


def main():
  race_track = ["########",
                "#NNNNNNE",
                "#NN#####",
                "#NN#####",
                "#SS#####"]

  race_track_obj = utils.Racetrack(race_track)

  behavior_policy = utils.behavior_policy(race_track_obj)

  policy_control(race_track_obj, behavior_policy)
  

main()