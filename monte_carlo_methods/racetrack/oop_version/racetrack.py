import numpy as np
import racetrack_utils as utils
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt

def incremental_prediction(Racetrack, episode, cum_IS, epsilon, gamma=1):

    episode.reverse()
    G = 0
    W = 1
    for step in range(len(episode) - 1):
        # G already accounts for reward -1 for each step
        step += 1
        G = (gamma * G) - 1
        x, y, v1, v2 = episode[step][0]  # state
        a1, a2 = episode[step][1]  # action
        cum_IS[x, y, v1, v2, a1, a2] += W

        print(
            f"Step T-{step} coordinate: ({x}, {y}, {v1}, {v2}) with action: ({a1}, {a2})"
        )

        # update q(s,a)
        next_state = utils.get_next_state(Racetrack, episode[step][0], episode[step][1])
        Racetrack.state_values[x][y][v1][v2] = round(Racetrack.get_state_value(next_state) 
            + ( (W / (cum_IS[x, y, v1, v2, a1, a2]))
            * (G - Racetrack.get_state_value(next_state)) # i think old is wrong following pseudo code (should be updating next state not current state value)
        ), 3) # NEW added rounding to 3 d.p for more accurate rounding

        # update target policy action based on best value of actions
        action_space_ls = utils.get_action_space(
            episode[step][0], Racetrack.racetrack[x][y]
        )
        # take optimal action according to current state values
        optimal_action_idx = utils.get_optimal_action(
            Racetrack, episode[step][0], action_space_ls
        )
        Racetrack.target_policy_dict[x][y][v1][v2] = action_space_ls[optimal_action_idx]

        print(f"  Action state value: {Racetrack.get_state_value(next_state)}")
        print(
            f"  Target policy action: {Racetrack.target_policy_dict[x][y][v1][v2]} with state value: {Racetrack.get_state_value(utils.get_next_state(Racetrack, episode[step][0], Racetrack.target_policy_dict[x][y][v1][v2]))}"
        )

        # compare if target_policy action matches current action taken
        if Racetrack.target_policy_dict[x][y][v1][v2] != episode[step][1]:
            return False

        W = W / ((1 - epsilon) + (epsilon / len(action_space_ls)))


def off_policy_control(
    Racetrack,
    epsilon=0.1,
    max_episode_count=10000,
    min_successful_episode=3,
    max_episode_generation_attempt=10,
    gamma=1,
):
    """
    max_episode_count: determines the maximum amount of successful episodes needed to be generated.
    min_successful_episode: determines the minimum amount of successful episodes generated for each starting positions.
    max_episode_generation_attempt: determines the maximum amount of ANY episode generated (success or failure).
    """
    # cumulative sum sampling ratio: state = 4d, action = 3d; total 7d
    cum_IS = np.zeros(
        (len(Racetrack.racetrack), len(Racetrack.racetrack[0]), 5, 5, 3, 3), int
    )
    success = True
    behaviour_policy = utils.get_policy(Racetrack, epsilon)
    episode_count = 0
    ep_generation_attempt = 0
    successful_ep_counter = [0 for _ in range(len(Racetrack.start_coord_list))]
    successful_epi_dict = dict(zip(Racetrack.start_coord_list, successful_ep_counter))
    episode = utils.Episode(Racetrack, behaviour_policy)
    while True:
        episode.policy = behaviour_policy
        episode.episode = []
        if episode.generate(Racetrack):
            if (
                incremental_prediction(
                    Racetrack, episode.episode, cum_IS, epsilon, gamma
                )
                == False
            ):
                behaviour_policy = utils.get_policy(Racetrack, epsilon)
                continue
            episode_count += 1  # successful attempt counter
            ep_generation_attempt += 1  # general attempt counter
            successful_epi_dict[
                (episode.episode[-1][0][0], episode.episode[-1][0][1])
            ] += 1
            print("Added one success attempt.")
            if min_successful_episode <= min(successful_epi_dict.values()) and episode_count >= max_episode_count:
                print(
                    "Off-policy control achieved minimum successful episodes for each starting location, ending run..."
                )
                print("Off-policy completed.")
                return success
            print(
                f"Off-policy control successful at episode {episode_count}, continuing... \n Smallest success value: {min(successful_epi_dict.values())}"
            )
            behaviour_policy = utils.get_policy(Racetrack, epsilon)
            ep_generation_attempt = 0
        else:
            """ep_generation_attempt += 1  # general attempt counter
            print(
                f"Episode generation reached max step, generation attempt {ep_generation_attempt} failed..."
            )
            if ep_generation_attempt > max_episode_generation_attempt:
                if epsilon < 1:
                    epsilon += 0.01
                    print(
                        f"Retrying with new behaviour policy with epsilon value of {round(epsilon, 2)} ...."
                    )
                else:
                    print(
                        "Epsilon value at maximum of 0.99, stopping off-policy control."
                    )
                    success = False
                    break
                ep_generation_attempt = 0"""
            behaviour_policy = utils.get_policy(Racetrack, epsilon)


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

def generate_routes_gif(Racetrack, race_track):
    episode = utils.Episode(Racetrack, Racetrack.target_policy_dict)
    track = build_track(race_track)
    os.chdir(f'reinforcement_learning_programs/monte_carlo_methods/racetrack/oop_version')
    for each_start in range(len(Racetrack.start_coord_list)):
        shutil.rmtree(f'racetrack_gifs/racetrack_{each_start}', ignore_errors=True)
        os.mkdir(f'racetrack_gifs/racetrack_{each_start}')
        images=[]
        episode.episode=[]
        episode.generate(Racetrack, start_pos=Racetrack.start_coord_list[each_start])
        for step in range(len(episode.episode)):
            track[len(Racetrack.racetrack[0]) - 1 - episode.episode[step][0][1]][episode.episode[step][0][0]] = 0.2
            plt.figure(figsize=(10, 10))
            plt.imshow(track)
            plt.title(f'Racetrack with start location {Racetrack.start_coord_list[each_start]}', fontsize=10)
            plt.savefig(f'racetrack_gifs/racetrack_{each_start}/Start-{each_start}-Step-{step}.png')
            image = Image.open(f'racetrack_gifs/racetrack_{each_start}/Start-{each_start}-Step-{step}.png')
            images.append(image)
            if race_track[len(Racetrack.racetrack[0]) - 1 - episode.episode[step][0][1]][episode.episode[step][0][0]] == "S":
                track[len(Racetrack.racetrack[0]) - 1 - episode.episode[step][0][1]][episode.episode[step][0][0]] = 0.6
            else:
                track[len(Racetrack.racetrack[0]) - 1 - episode.episode[step][0][1]][episode.episode[step][0][0]] = 1
        images[0].save(f'racetrack_gifs/racetrack_{each_start}/Optimal_path_for_{Racetrack.start_coord_list[each_start]}.gif', save_all=True, append_images=images[1:], duration=200, loop=0)

def main():
    race_track = ["#######E", 
                  "#NNNNNNE", 
                  "#NNNNNNE", 
                  "#NNNNNNE", 
                  "#SS#####"]

    epsilon = 0.1
    gamma = 0.9
    max_episode_count = 100
    max_episode_generation_attempt = 100000
    race_track_obj = utils.Racetrack(race_track)

    MC_control_result = off_policy_control(
        race_track_obj,
        epsilon=epsilon,
        max_episode_count=max_episode_count,
        max_episode_generation_attempt=max_episode_generation_attempt,
        gamma=gamma,
    )

    if MC_control_result == True:
        generate_routes_gif(race_track_obj, race_track)
    else:
        print("Policy failed, ending algorithm.")

if __name__ == "__main__":
    main()

