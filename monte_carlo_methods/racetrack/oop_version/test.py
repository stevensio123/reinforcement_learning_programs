import racetrack_utils as utils
import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt

race_track = ["####EEEE",
              "#NNNNNNE",
              "#NNNNNNE",
              "#NNNNNNE",
              "#NNNNN##",
              "#SSSSS##"]

verts = [(2,4), (3,3), (4,2), (7,1)]

codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]

def build_track(og_racetrack):
    racetrack = np.array([list(row) for row in og_racetrack])
    track = np.ones(shape=(len(racetrack),len(racetrack[0])))
    for row in range(len(racetrack)):
        for column in range(len(racetrack[row])):
            if racetrack[row][column] == "#":
                track[row][column] = 0
            elif racetrack[row][column] == "E":
                track[row][column] = 0.4
            elif racetrack[row][column] == "S":
                track[row][column] = 0.6
    
    return track

print("\nTEST: racetrack object")
racetrack = utils.Racetrack(race_track)
print(f"racetrack list (cartesian):{racetrack.racetrack}")
print(f"start locations: {racetrack.start_coord_list}")
print(f"terminal locations: {racetrack.terminal_coord_list}")
print(f"racetrack[7][3]: {racetrack.racetrack[7][3]}")
print(f"policy[7][3][1][2]: {racetrack.target_policy_dict[7][3][1][2]}")


print("\nTEST: next_state function")
state = (0,2,0,0)
# state = (0,7,0,0) # out of bounds
action = [1,1]
print(f"Current state: {state}")
print(f"Action taken: {action}")
try:
    x, y , vx, vy = utils.get_next_state(racetrack, state, action)
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
policy = utils.get_policy(racetrack, epsilon=0.4)
print(f"Policy shape: {policy.shape}")
print(f"Policy for (1,0,0,0): {policy[1][0][0][0]}")
print(f"Policy for (2,0,0,0): {policy[2][0][0][0]}")
print(f"Policy for (0,2,4,4): {policy[0][2][4][4]}")

print("\nTEST: Episode class")
episode = utils.Episode(racetrack, policy)
episode.generate(racetrack)
print(episode)


successful_ep_counter = [ 0 for _ in range(len(racetrack.start_coord_list))]
print(successful_ep_counter)
print(racetrack.start_coord_list)
successful_epi_dict = dict(zip(racetrack.start_coord_list, successful_ep_counter))
print(successful_epi_dict)

track = build_track(race_track)
path = Path(verts, codes)
patch = patches.PathPatch(path, facecolor="none", lw=2)
plt.figure(figsize=(10, 10))
plt.imshow(track)
plt.gca().add_patch(patch)
plt.title("Racetrack")
plt.show()