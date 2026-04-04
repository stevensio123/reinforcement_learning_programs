import racetrack_utils as utils
import numpy as np
race_track = ["########",
              "#NNNNNNE",
              "#NN#####",
              "#NN#####",
              "#SS#####"]

print("\nTEST: racetrack object")
racetrack = utils.Racetrack(race_track)
print(f"racetrack list (cartesian):{racetrack.racetrack}")
print(f"start locations: {racetrack.start_coord_list}")
print(f"terminal locations: {racetrack.terminal_coord_list}")

print(f"racetrack[7][3]: {racetrack.racetrack[7][3]}")


print("\nTEST: next_state function")
state = (0,2,0,0)
# state = (0,7,0,0) # out of bounds
action = [3,3]
print(f"Current state: {state}")
print(f"Action taken: {action}")
try:
    x, y , vx, vy = utils.get_next_state(racetrack.racetrack, state, action)
    print(f"Next state: ({x}, {y}, {vx}, {vy})")
except TypeError:
    print("Crashed or out of bounds")


print("\nTEST: action space function")
state = (0,2,0,0)
print(f"Action space for {state}: {utils.get_action_space(state)}")
state = (0,2,4,4)
print(f"Action space for {state}: {utils.get_action_space(state)}")

print("\nTEST: state space class")
state_space = utils.StateSpace(race_track)
print(f"State values shape: {state_space.state_values.shape}")
print(f"State value for (0,2,0,0): {state_space.get_state_value((0,2,0,0))}")

print("\nTEST: behavior policy function")
policy = utils.behavior_policy(state_space, epsilon=0.2)
print(f"Policy shape: {policy.shape}")
print(f"Policy for (0,2,0,0): {policy[0][2][0][0]}")
print(f"Policy for (0,2,4,4): {policy[0][2][4][4]}")

print("\nTEST: Episode class")
episode = utils.Episode(race_track, policy)
episode.generate()
print(f"Episode steps: {episode.steps}")
print(f"Episode trajectory: {episode.episode}")
print(episode)
