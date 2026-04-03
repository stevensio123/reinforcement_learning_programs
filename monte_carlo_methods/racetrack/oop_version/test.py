import racetrack_classes
import numpy as np
race_track = ["########",
              "#NNNNNNE",
              "#NN#####",
              "#NN#####",
              "#SS#####"]
racetrack = racetrack_classes.racetrack(race_track)
print(racetrack.racetrack)
print(racetrack.start_locs())
print(racetrack.terminal_locs())
print(racetrack.racetrack[7][1])
