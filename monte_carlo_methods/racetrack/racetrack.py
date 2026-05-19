import logging
import numpy as np
from tqdm import tqdm
import racetrack_utils as utils

# format for tqdm progress bar
pbar_format = "{desc:<40}[{bar:50}] {percentage:3.2f}% | ETA {remaining}"


logger = logging.getLogger(__name__)


def incremental_prediction(Racetrack, episode, cum_is, epsilon, gamma=0.9):
    """
    Racetrack : Racetrack class from utils.
    episode : episode list generated from Episode class in utils.
    cum_is : the current cumulative importance sampling ratio in the policy control iteration.
    epsilon : prob. of choosing random action. Used to compute b(A|S), the probability of action given state in the behaviour policy.
    gamma : the weight parameter used on the rewards.
    """
    logger.debug(
        "starting incremental prediction for episode with length %d", len(episode)
    )
    episode.reverse()
    g = 0
    w = 1
    for step in range(len(episode)):
        # g already accounts for reward -1 for each step
        g = (gamma * g) - 1
        # state and action of step T-1 of the episode
        x, y, v1, v2 = episode[step][0]  # state
        a1, a2 = episode[step][1]  # action
        cum_is[x, y, v1, v2, a1, a2] += w

        logger.debug(
            "step T-%d coordinate: (%d, %d, %d, %d) with action: (%d, %d)",
            step + 1,  # so it writes T-1 for the last instead of T-0
            x,
            y,
            v1,
            v2,
            a1,
            a2,
        )

        # update q(s,a) using:
        # q(s,a)' = q(s,a) + W/C(s,a) * (G - q(s,a))
        Racetrack.action_values[x, y, v1, v2, a1, a2] = round(
            Racetrack.get_action_value(episode[step][0], episode[step][1])
            + (
                (w / (cum_is[x, y, v1, v2, a1, a2]))
                * (g - Racetrack.get_action_value(episode[step][0], episode[step][1]))
            ),
            2,
        )
        """
        update target policy action based on best value of actions
        take optimal action according to current state values, but
        take episode's actual action if it is one of the optimal actions (in case of draws).
        """
        action_space_ls = utils.get_action_space(episode[step][0])
        optimal_action_idx = utils.get_optimal_action(
            Racetrack, episode[step][0], action_space_ls, episode[step][1]
        )
        Racetrack.target_policy_dict[x][y][v1][v2] = action_space_ls[optimal_action_idx]

        # compare if target_policy action matches current action taken
        if Racetrack.target_policy_dict[x][y][v1][v2] != episode[step][1]:
            logger.debug(
                "incremental prediction Failure: behaviour policy action and target policy action at step T-%d mismatch:  current action (%d, %d) value = %.3f | expected action (%d, %d) value = %.3f",
                step,
                episode[step][1][0],  # episode action
                episode[step][1][1],  # ''
                Racetrack.get_action_value(
                    episode[step][0], episode[step][1]
                ),  # episode action value
                Racetrack.target_policy_dict[x][y][v1][v2][0],  # target policy action
                Racetrack.target_policy_dict[x][y][v1][v2][1],  # ''
                Racetrack.get_action_value(
                    episode[step][0], Racetrack.target_policy_dict[x][y][v1][v2]
                ),  # target policy action value
            )
            return False

        # update importance sampling ratio
        w = w / ((1 - epsilon) + (epsilon / len(action_space_ls)))
        logger.debug("importance sampling weight W = %d at step T-%d", w, step)

    return True


def off_policy_control(
    Racetrack,
    epsilon=0.1,
    minimum_episode_requirement=1000,
    minimum_starts_requirement=100,
    epsilon_update=False,
    gamma=0.9,
):
    """
    Racetrack: Racetrack object from utils.
    epsilon: the starting epsilon value for soft policy.
    minimum_episode_requirement: The minimum amount of episodes generated.
    minimum_starts_requirement: The minimum amount of episodes generated for any starting states.
    epsilon_update: If set to True, increase epsilon by 0.01 when failed episode generation attempts reaches 100K, resets to zero if there is one success.
    gamma: The discounting factor used on rewards.
    """
    # cumulative sum importance sampling ratio: state = 4d, action = 3d; total 7d
    cum_is = np.zeros(
        (len(Racetrack.racetrack), len(Racetrack.racetrack[0]), 5, 5, 3, 3), int
    )

    attempted_ep_counter = 0  # count for any attempt to generate episode
    failed_attempt_ep_counter = 0  # failed episodes for attempted episode generation
    generated_ep_counter = (
        0  # count for any ep generated, success or not in incremental prediction
    )
    success_ep_counter = 0  # successful episode for incremental prediction (only when no mismatch from step 0 to T-1)

    success_ep_state_counter = [0 for _ in range(len(Racetrack.start_coord_list))]
    successful_epi_dict = dict(
        zip(Racetrack.start_coord_list, success_ep_state_counter)
    )
    Episode = utils.Episode(Racetrack)
    logger.info("starting off-policy control")

    # define progress bar for epsilon value
    pbar_epsilon = tqdm(
        total=1,
        desc="current epsilon value for policy in episode generation",
        position=1,
        leave=True,
        bar_format=pbar_format,
    )
    # start bar at initialized epsilon, scaled to 100 for better updates
    pbar_epsilon.n = epsilon
    pbar_epsilon.refresh()

    # define progress bar for total episodes generated count
    pbar_overall = tqdm(
        total=minimum_episode_requirement,
        desc="episodes generated",
        position=2,
        leave=True,
        bar_format=pbar_format,
    )

    # define progress bars for successful episode's start state count, bars are a dict, so values are accessed from the key (state coord)
    pbar_starts = {
        coord: tqdm(
            total=minimum_starts_requirement,
            desc=f"episode success for start coord {coord}",
            position=i
            + 3,  # +3 because failed attempts, epsilon and inc.pred. successes take up position 0, 1, 2
            leave=True,
            bar_format=pbar_format,
        )
        for i, coord in enumerate(Racetrack.start_coord_list)
    }

    # helper function to close all pbars after algorithm ends
    def close_pbars():
        pbar_epsilon.close()
        pbar_overall.close()
        for pbar in pbar_starts.values():
            pbar.close()

    while True:
        Episode.episode = []  # reset episode list
        if Episode.generate(Racetrack, epsilon):
            failed_attempt_ep_counter = 0  # reset in case of a success
            attempted_ep_counter += 1
            generated_ep_counter += 1
            success_coord = (Episode.episode[0][0][0], Episode.episode[0][0][1])
            # counter for starting state for this successful episode
            successful_epi_dict[(success_coord)] += 1
            if pbar_starts[(success_coord)].n < minimum_starts_requirement:
                pbar_starts[(success_coord)].update(1)

            if pbar_overall.n < minimum_episode_requirement:
                pbar_overall.update(1)

            if incremental_prediction(
                Racetrack, Episode.episode, cum_is, epsilon, gamma
            ):
                success_ep_counter += 1  # successful ep for incremental prediction
                logger.info(
                    "Incremental prediction successful, %d successful incremental prediction runs so far...",
                    success_ep_counter,
                )
            else:
                logger.info(
                    "Incremental prediction failed at generation attempt. Retrying..."
                )

            """
            - episode generation is successful, also ran incremental prediction (success / failure)
            - now check if success ep count for each start state reached max
            - update behavior policy with new epsilon if failed ep count reached max
            - reset failure ep counter and bar for new epsilon-soft policy
            """
            if (generated_ep_counter >= minimum_episode_requirement) and (
                min(successful_epi_dict.values()) >= minimum_starts_requirement
            ):
                logger.info(
                    "off-policy control achieved minimum successful episodes for each starting states, ending run..."
                )
                logger.info(
                    "off-policy control completed after:\n %d successful episodes generated and\n %d total episode generation attempts with current epsilon value %.2f",
                    generated_ep_counter,
                    attempted_ep_counter,
                    epsilon,
                )
                close_pbars()
                return True

            else:
                continue

        else:
            """
            unsuccessful episode generation, either due to reaching max step count or failure in generating episode.
                - if failed attempt counter reached 100K, update epsilon if epsilon_update=True, 
                    - reset failed attempt counter and bar
                - else, just update behaviour policy and try again
            """
            attempted_ep_counter += 1
            failed_attempt_ep_counter += 1
            if failed_attempt_ep_counter > 100000:
                logger.debug(
                    "episode generation failed over the maximum of 1M times with current epsilon value %.2f",
                    epsilon,
                )
                if epsilon_update:
                    if epsilon < 1.00:
                        epsilon += 0.01
                        pbar_epsilon.update(0.01)
                        logger.debug("Updated epsilon value to %.2f.", epsilon)
                        failed_attempt_ep_counter = 0
                    else:
                        logger.debug(
                            "epsilon value reached maximum of 1, stopping off-policy control."
                        )
                        tqdm.write("something is wrong.")
                        close_pbars()
                        return False


def main():
    race_track = [
        "####NNNNNNNNNNNNNNE",
        "####NNNNNNNNNNNNNNE",
        "###NNNNNNNNNNNNNNNE",
        "###NNNNNNNNNNNNNNNE",
        "##NNNNNNNNNNNNNNNNE",
        "##NNNNNNNNNNNNNNNNE",
        "#NNNNNNNNNNNNNNNNNE",
        "#NNNNNNNNNNNNNNNNNE",
        "NNNNNNNNNNNN#######",
        "NNNNNNNNNNNN#######",
        "NNNNNNNNNN#########",
        "NNNNNNNNNN#########",
        "NNNNNNNNNN#########",
        "NNNNNNNNNN#########",
        "#NNNNNNNNN#########",
        "#NNNNNNNNN#########",
        "#NNNNNNNNN#########",
        "#NNNNNNNNN#########",
        "#NNNNNNNNN#########",
        "#NNNNNNNNN#########",
        "##NNNNNNNN#########",
        "##NNNNNNNN#########",
        "##NNNNNNNN#########",
        "##NNNNNNNN#########",
        "##NNNNNNNN#########",
        "##NNNNNNNN#########",
        "###NNNNNNN#########",
        "###NNNNNNN#########",
        "####SSSSSS#########",
    ]

    race_track_obj = utils.Racetrack(race_track)
    max_MC_control_attempt = 0

    while True:
        max_MC_control_attempt += 1
        if max_MC_control_attempt > 5:
            break
        MC_control_result = off_policy_control(
            Racetrack=race_track_obj,
            epsilon=0.25,
            minimum_episode_requirement=120000,
            minimum_starts_requirement=20000,
            gamma=0.9,
        )

        if MC_control_result:
            if utils.generate_routes_gif(race_track_obj, race_track):
                logger.info("Gifs generated.")
            return True
        else:
            logger.debug("Policy Control failed for this attempt.")


if __name__ == "__main__":
    main()
