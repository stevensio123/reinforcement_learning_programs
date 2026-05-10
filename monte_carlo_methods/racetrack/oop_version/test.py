import racetrack_utils as utils
import racetrack as rtck
import shutil
import PIL as Image
import numpy as np
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

'''
verts = [(2,4), (3,3), (4,2), (7,1)]

codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]

#Required to create line paths in plot drawings for matplotlib.path
'''

"""def generate_routes_gif(Racetrack, start_pos):
    os.chdir(r'\racetrack_gifs')
    episode = utils.Episode(Racetrack, Racetrack.target_policy_dict)
    track = rtck.build_track(Racetrack)
    shutil.rmtree(rf'\racetrack1', ignore_errors=True)
    os.makedirs(rf'\racetrack1', exist_ok=True)
    os.chdir(rf'\racetrack1')
    images=[]
    episode.generate(Racetrack, start_pos=start_pos)
    for step in range(len(episode.episode)):
        track[episode.episode[step][0][0]][episode.episode[step][0][1]] = 0.5
        plt.figure(figsize=(10, 10))
        plt.imshow(track)
        plt.title(f"Racetrack with start location {start_pos}", fontsize=10)
        plt.savefig(f"Step_{step}.png")
        image = Image.open(f"Step_{step}.png")
        images.append(image)
        track[episode.episode[step][0][0]][episode.episode[step][0][1]] = 1
    images[0].save(f"Optimal_path_for_{start_pos}.gif", save_all=True, append_images=[images[1]], duration=200, loop=0)
    os.chdir('..')
    os.chdir('..')"""

def main():
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

    """generate_routes_gif(racetrack, episode.episode[0])"""

    """
    track = rtck.build_track(race_track)

    '''
    To create a path, the turtle just moves following the Path function but the patches.PathPatch creates the line that is visible
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor="none", lw=2)
    #'''

    plt.figure(figsize=(10, 10))
    plt.imshow(track)

    '''
    To add the path to the figure
    plt.gca().add_patch(patch)
    '''

    plt.title("Racetrack")
    #plt.show()

    print(os.getcwd())
    """

if __name__ == "__main__":
    main()
