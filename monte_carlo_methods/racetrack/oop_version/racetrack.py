import numpy as np
import racetrack_utils as utils
import logging

logger = logging.getlogger(__name__)
logger.setlevel(logging.DEBUG)

# set file handlers for each level
debug_handler = utils.set_file_handler("debug.log", logging.DEBUG)
info_handler = utils.set_file_handler("info.log", logging.INFO)
warn_handler = utils.set_file_handler("warning.log", logging.WARNING)
error_handler = utils.set_file_handler("error.log", logging.ERROR)
logger.addhandler(debug_handler)
logger.addhandler(info_handler)
logger.addhandler(warn_handler)
logger.addhandler(error_handler)


def incremental_prediction(racetrack, episode, cum_is, epsilon, gamma=0.9):
    """
    racetrack : Racetrack class from utils.
    episode : Episode class from utils, expects episode to be generated.

    """
    episode.reverse()
    g = 0
    w = 1
    for step in range(len(episode) - 1):
        # g already accounts for reward -1 for each step
        step += 1
        g = (gamma * g) - 1
        x, y, v1, v2 = episode[step][0]  # state
        a1, a2 = episode[step][1]  # action
        cum_is[x, y, v1, v2, a1, a2] += w

        print(
            f"step t-{step} coordinate: ({x}, {y}, {v1}, {v2}) with action: ({a1}, {a2})"
        )

        # update q(s,a)
        next_state = utils.get_next_state(racetrack, episode[step][0], episode[step][1])
        racetrack.state_values[x][y][v1][v2] = racetrack.get_state_value(next_state) + (
            (w / (cum_is[x, y, v1, v2, a1, a2]))
            * (g - racetrack.get_state_value(next_state))
        )

        # update target policy action based on best value of actions
        action_space_ls = utils.get_action_space(
            episode[step][0], racetrack.racetrack[x][y]
        )
        # take optimal action according to current state values
        optimal_action_idx = utils.get_optimal_action(
            racetrack, episode[step][0], action_space_ls
        )
        racetrack.target_policy_dict[x][y][v1][v2] = action_space_ls[optimal_action_idx]

        print(f"  action state value: {racetrack.get_state_value(next_state)}")
        print(
            f"  target policy action: {racetrack.target_policy_dict[x][y][v1][v2]} with state value: {racetrack.get_state_value(utils.get_next_state(racetrack, episode[step][0], racetrack.target_policy_dict[x][y][v1][v2]))}"
        )

        # compare if target_policy action matches current action taken
        if racetrack.target_policy_dict[x][y][v1][v2] != episode[step][1]:
            return false

        w = w / ((1 - epsilon) + (epsilon / len(action_space_ls)))


def off_policy_control(
    racetrack,
    epsilon=0.1,
    max_episode_count=10000,
    min_successful_episode=3,
    max_episode_generation_attempt=10,
    gamma=0.9,
):
    """
    max_episode_count: determines the maximum amount of successful episodes needed to be generated.
    min_successful_episode: determines the minimum amount of successful episodes generated for each starting positions.
    max_episode_generation_attempt: determines the maximum amount of any episode generated (success or failure).
    """
    # cumulative sum sampling ratio: state = 4d, action = 3d; total 7d
    cum_is = np.zeros(
        (len(racetrack.racetrack), len(racetrack.racetrack[0]), 5, 5, 3, 3), int
    )
    behaviour_policy = utils.get_policy(racetrack, epsilon)
    episode_count = 0
    ep_generation_attempt = 0
    successful_ep_counter = [0 for _ in range(len(racetrack.start_coord_list))]
    successful_epi_dict = dict(zip(racetrack.start_coord_list, successful_ep_counter))
    episode = utils.episode(racetrack, behaviour_policy)
    while true:
        episode.policy = behaviour_policy
        episode.episode = []
        if episode.generate(racetrack):
            if (
                incremental_prediction(
                    racetrack, episode.episode, cum_is, epsilon, gamma
                )
                == false
            ):
                behaviour_policy = utils.get_policy(racetrack, epsilon)
                continue
            episode_count += 1  # successful attempt counter
            ep_generation_attempt += 1  # general attempt counter
            successful_epi_dict[
                (episode.episode[-1][0][0], episode.episode[-1][0][1])
            ] += 1
            if (min(successful_epi_dict.values()) >= min_successful_episode) and (
                episode_count >= max_episode_count
            ):
                print(
                    "off-policy control achieved minimum successful episodes for each starting location, ending run..."
                )
                break
            print(
                f"off-policy control successful at episode {episode_count}, continuing..."
            )
        else:
            ep_generation_attempt += 1  # general attempt counter
            print(
                f"episode generation reached max step, generation attempt {ep_generation_attempt} failed..."
            )
            if ep_generation_attempt > max_episode_generation_attempt:
                if epsilon < 1:
                    epsilon += 0.01
                    print(
                        f"retrying with new behaviour policy with epsilon value of {round(epsilon, 2)} ...."
                    )
                else:
                    print(
                        "epsilon value at maximum of 0.99, stopping off-policy control."
                    )
                    break
                ep_generation_attempt = 0
            behaviour_policy = utils.get_policy(racetrack, epsilon)

    print("off-policy completed.")


def main():
    race_track = ["#######e", "#nnnnnne", "#nnnnnne", "#nnnnnne", "#ss#####"]

    epsilon = 0.1
    gamma = 0.9
    max_episode_count = 1000
    max_episode_generation_attempt = 4
    race_track_obj = utils.racetrack(race_track)

    off_policy_control(
        race_track_obj,
        epsilon,
        max_episode_count,
        max_episode_generation_attempt,
        gamma,
    )


main()
