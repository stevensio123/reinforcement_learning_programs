import racetrack_utils as utils
import numpy as np
race_track = ["########",
              "#NNNNNNE",
              "#NN#####",
              "#NN#####",
              "#SS#####"]

# TEST: racetrack object
racetrack = utils.Racetrack(race_track)
print(racetrack.racetrack)
print(racetrack.start_locs())
print(racetrack.terminal_locs())
print(racetrack.racetrack[7][3])


# TEST: next_state function
state = (0,2,0,0)
# state = (0,7,0,0) # out of bounds
action = [1,1]
try:
    x, y , vx, vy = utils.next_state(racetrack.racetrack, state, action)
    print(x, y, vx, vy)
except TypeError:
    print("Crashed or out of bounds")


