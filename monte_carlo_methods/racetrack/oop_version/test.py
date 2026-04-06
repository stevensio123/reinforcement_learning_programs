import racetrack_utils as utils
import numpy as np
race_track = ["####EEEE",
              "#NNNNNNE",
              "#NNNNNNE",
              "#NNNNNNE",
              "#NNNNN##",
              "#SSSSS##"]

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
print(f"State values shape: {racetrack.state_values.shape}")
print(f"State value for (1,0,0,0): {racetrack.get_state_value((1,0,0,0))}")

print("\nTEST: behavior policy function")
policy = utils.behavior_policy(racetrack, epsilon=0.4)
print(f"Policy shape: {policy.shape}")
print(f"Policy for (1,0,0,0): {policy[1][0][0][0]}")
print(f"Policy for (2,0,0,0): {policy[2][0][0][0]}")
print(f"Policy for (0,2,4,4): {policy[0][2][4][4]}")

print("\nTEST: Episode class")
episode = utils.Episode(racetrack, policy)
episode.generate(racetrack)
print(episode)
