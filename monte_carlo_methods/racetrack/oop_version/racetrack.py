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
        # step += 1   # no need to skip it since episode generation did not include T step
        g = (gamma * g) - 1
        x, y, v1, v2 = episode[step][0]  # state
        a1, a2 = episode[step][1]  # action
        cum_is[x, y, v1, v2, a1, a2] += w

        logger.debug(
            "step T-%d coordinate: (%d, %d, %d, %d) with action: (%d, %d)",
            step,
            x,
            y,
            v1,
            v2,
            a1,
            a2,
        )

        # update q(s,a) using estimated value of target policy
        Racetrack.action_values[x, y, v1, v2, a1, a2] = round(
            Racetrack.get_action_value(episode[step][0], episode[step][1])
            + (
                (w / (cum_is[x, y, v1, v2, a1, a2]))
                * (g - Racetrack.get_action_value(episode[step][0], episode[step][1]))
            ),
            3,
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
            # instead of return False and go to next episode, why not continue to next step and continue prediction?
            # so if we go from T-1 (mismatch step) to T-2 (match step), then we update w, but we would not update w otherwise
            break

        # update importance sampling ratio
        w = w / ((1 - epsilon) + (epsilon / len(action_space_ls)))
        logger.debug("importance sampling weight W = %d at step T-%d", w, step)

    return True


def off_policy_control(
    Racetrack,
    epsilon,
    max_successful_episode,
    min_successful_episode,
    max_failed_episode_generation_attempt,
    gamma,
):
    """
    Racetrack: Racetrack object from utils.
    max_successful_episode: determines the maximum amount of successful episodes needed to be generated.
    min_successful_episode: determines the minimum amount of successful episodes generated for each starting positions.
    max_episode_generation_attempt: determines the maximum amount of any episode generated (success or failure).
    """
    # cumulative sum sampling ratio: state = 4d, action = 3d; total 7d
    cum_is = np.zeros(
        (len(Racetrack.racetrack), len(Racetrack.racetrack[0]), 5, 5, 3, 3), int
    )
    behaviour_policy = utils.get_policy(Racetrack, epsilon)

    attempted_ep_counter = 0  # count for any attempt to generate episode
    failed_attempt_ep_counter = 0  # failed episodes for attempted episode generation
    generated_ep_counter = (
        0  # count for any ep generated, success or not in incremental prediction
    )
    success_ep_counter = 0  # successful episode for incremental prediction

    success_ep_state_counter = [0 for _ in range(len(Racetrack.start_coord_list))]
    successful_epi_dict = dict(
        zip(Racetrack.start_coord_list, success_ep_state_counter)
    )
    Episode = utils.Episode(Racetrack, behaviour_policy)
    logger.info("starting off-policy control")
    # tqdm.write("starting off-policy control")

    # define progress bar for failed generation attempts value
    pbar_failed_attempt_ep_count = tqdm(
        total=max_failed_episode_generation_attempt,
        desc="failed episode generation attempts (resets for any new policy)",
        position=0,
        leave=True,
        bar_format=pbar_format,
        mininterval=1,
    )

    # define progress bar for epsilon value
    pbar_epsilon = tqdm(
        total=1,
        desc="current epsilon value for policy regeneration",
        position=1,
        leave=True,
        bar_format=pbar_format,
    )
    # start bar at initialized epsilon, scaled to 100 for better updates
    pbar_epsilon.n = epsilon
    pbar_epsilon.refresh()

    # define progress bar for successful episode count
    pbar_overall = tqdm(
        total=max_successful_episode,
        desc="successful incremental prediction runs",
        position=2,
        leave=True,
        bar_format=pbar_format,
    )

    # define progress bars for successful episode's start state count, bars are a dict, so values are accessed from the key (state coord)
    pbar_starts = {
        coord: tqdm(
            total=min_successful_episode,
            desc=f"episode success for start coord {coord}",
            # mininterval=1,
            position=i+ 3,  # +3 because failed attempts, epsilon and inc.pred. successes take up position 0, 1, 2
            leave=True,
            bar_format=pbar_format,
        )
        for i, coord in enumerate(Racetrack.start_coord_list)
    }

    while True:
        Episode.policy = behaviour_policy
        Episode.episode = []  # reset episode list
        if Episode.generate(Racetrack) and (
            failed_attempt_ep_counter < max_failed_episode_generation_attempt
        ):
            attempted_ep_counter += 1
            generated_ep_counter += 1
            if incremental_prediction(
                Racetrack, Episode.episode, cum_is, epsilon, gamma
            ):
                success_ep_counter += 1  # successful ep for incremental prediction
                logger.info("Incremental prediction successful")
                pbar_overall.update(1)  # update for pbar too
                success_coord = (Episode.episode[-1][0][0], Episode.episode[-1][0][1])
                # counter for starting state for this successful episode
                successful_epi_dict[(success_coord)] += 1
                if pbar_starts[(success_coord)].n < min_successful_episode:
                    pbar_starts[(success_coord)].update(1)

            else:
                logger.info(
                    "Incremental prediction failed at generation attempt %d. Retrying with updated policy...",
                    failed_attempt_ep_counter,
                )
                behaviour_policy = utils.get_policy(Racetrack, epsilon)
                # pbar_failed_attempt_ep_count.refresh()
                continue

            """
            - episode generation is successful, also ran incremental prediction (success / failure)
            - now check if success ep count for each start state reached max
            - update behavior policy with new epsilon if failed ep count reached max
            - reset failure ep counter and bar for new policy
            """
            if (min(successful_epi_dict.values()) >= min_successful_episode) and (
                success_ep_counter >= max_successful_episode
            ):
                logger.info(
                    "off-policy control achieved minimum successful episodes for each starting location, ending run..."
                )
                logger.info(
                    "Off-policy control successful after %d successful episodes generated with current epsilon value %.2f",
                    success_ep_counter,
                    epsilon,
                )
                return True

            behaviour_policy = utils.get_policy(Racetrack, epsilon)
            failed_attempt_ep_counter = 0
            pbar_failed_attempt_ep_count.reset()
            pbar_failed_attempt_ep_count.refresh()
        else:
            """
            unsuccessful episode generation, either due to reaching max step count or failure in generating episode.
                - if failed attempt counter reached max, update epsilon and behaviour policy, 
                    - reset failed attempt counter and bar
                - else, just update behaviour policy and try again
            """
            attempted_ep_counter += 1
            failed_attempt_ep_counter += 1
            pbar_failed_attempt_ep_count.update(1)
            if failed_attempt_ep_counter > max_failed_episode_generation_attempt:
                logger.debug(
                    "episode generation failed over the maximum of %d times with current epsilon value %.2f",
                    max_failed_episode_generation_attempt,
                    epsilon,
                )
                if epsilon < 1.00:
                    epsilon += 0.01
                    pbar_epsilon.update(0.01)
                    logger.debug("Updated epsilon value to %.2f.", epsilon)
                else:
                    logger.debug(
                        "epsilon value reached maximum of 1, stopping off-policy control."
                    )
                    tqdm.write("something is wrong.")
                    return False
                failed_attempt_ep_counter = 0
                pbar_failed_attempt_ep_count.reset()
                pbar_failed_attempt_ep_count.refresh()

            # logger.debug("creating new behaviour policy with epsilon %.2f", epsilon)
            behaviour_policy = utils.get_policy(Racetrack, epsilon)


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

    MC_control_result = off_policy_control(
        Racetrack=race_track_obj,
        epsilon=0.1,
        max_successful_episode=10000,
        min_successful_episode=10,
        max_failed_episode_generation_attempt=1000,
        gamma=0.9,
    )

    if MC_control_result:
        utils.generate_routes_gif(race_track_obj, race_track)
    else:
        tqdm.write("Policy failed, ending algorithm.")


if __name__ == "__main__":
    main()
