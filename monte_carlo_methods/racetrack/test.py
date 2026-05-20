import racetrack_utils as utils
# just for testing functions in utils

"""
verts = [(2,4), (3,3), (4,2), (7,1)]

codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]

#Required to create line paths in plot drawings for matplotlib.path
"""

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

race_track = [
    "##NNNNNNNNEEE######",
    "##NNNNNNNNEEE######",
    "##NNNNNNNN#########",
    "###NNNNNNN#########",
    "###NNNNNNN#########",
    "####SSSSSS#########",
]


csv_file = input("CSV file name? ")
if csv_file.endswith(".csv"):
    try:
        imported_file, max_steps, min_steps = utils.import_csv(csv_file)
    except:
        print("File not found")
        raise SystemExit
    print("\nTEST: racetrack object")
    racetrack = utils.Racetrack(imported_file, min_steps, max_steps)
    print(f"racetrack list (cartesian):{racetrack.racetrack}")
    print(f"start locations: {racetrack.start_coord_list}")
    print(f"terminal locations: {racetrack.terminal_coord_list}")
    print(f"racetrack[7][3]: {racetrack.racetrack[7][3]}")
    print(f"policy[7][3][1][2]: {racetrack.target_policy_dict[7][3][1][2]}")


    print("\nTEST: next_state function")
    state = (0, 2, 0, 0)
    # state = (0,7,0,0) # out of bounds
    action = [1, 1]
    print(f"Current state: {state}")
    print(f"Action taken: {action}")
    try:
        x, y, vx, vy = utils.get_next_state(racetrack, state, action)
        print(f"Next state: ({x}, {y}, {vx}, {vy})")
    except TypeError:
        print("Crashed or out of bounds")


    print("\nTEST: action space function")
    state = (0, 2, 0, 0)
    action_ls_1 = utils.get_action_space(state)
    print(f"Action space for {state}: {action_ls_1}")
    print(
        f"Optimal action for {state}: {utils.get_optimal_action(racetrack, state, action_ls_1)}"
    )
    state = (0, 2, 4, 4)
    action_ls_1 = utils.get_action_space(state)
    print(f"Action space for {state}: {action_ls_1}")
    print(
        f"Optimal action for {state}: {utils.get_optimal_action(racetrack, state, action_ls_1, step_action=[-1, 0])}"
    )

else:
    print("File is not the correct format")
