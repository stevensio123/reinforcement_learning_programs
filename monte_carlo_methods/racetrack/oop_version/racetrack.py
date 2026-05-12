import logging
import logging.config
from pathlib import Path

log_dir = Path(__file__).resolve().parent

formatter = logging.Formatter()
logging.config.dictConfig(
    {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(name)s - %(levelname)s | %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "WARNING",  # setting to INFO / DEBUG causes conflicts with displaying tqdm progress bars
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": f"{log_dir}/racetrack.log",
                "formatter": "standard",
                "level": "DEBUG",
                "mode": "w",  # write or replace log file
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],  # logs to both console and file
        },
        # this part only needed if want to customize logger in utils
        "loggers": {       
            utils.__name__: {
              "level": "INFO", # set to INFO to avoid too much logs from utils
          }
        }
    }
)

import numpy as np
from tqdm import tqdm
import racetrack_utils as utils

# format for tqdm progress bar
pbar_format = "{desc:<40}[{bar:50}] {percentage:3.0f}% | ETA {remaining}"


logger = logging.getLogger(__name__)

def incremental_prediction(Racetrack, episode, cum_is, epsilon, gamma=0.9):
    """
    Racetrack : Racetrack class from utils.
    episode : episode list generated from Episode class in utils.
    cum_is : the current cumulative importance sampling ratio in the policy control iteration.
    epsilon : prob. of choosing random action. Used to compute b(A|S), the probability of action given state in the behaviour policy.
    gamma : the weight parameter used on the rewards.
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

        logger.debug(
            "step t-%d coordinate: (%d, %d, %d, %d) with action: (%d, %d)",
            step,
            x,
            y,
            v1,
            v2,
            a1,
            a2,
        )

        # update q(s,a) using estimated value of target policy
        # get next state using state and action of current step
        next_state = utils.get_next_state(Racetrack, episode[step][0], episode[step][1])
        Racetrack.state_values[next_state] = round(
            Racetrack.get_state_value(next_state)
            + (
                (w / (cum_is[x, y, v1, v2, a1, a2]))
                * (
                    g - Racetrack.get_state_value(next_state)
                )  # i think old is wrong following pseudo code (should be updating next state not current state value)
            ),
            3,
        )  # NEW added rounding to 3 d.p for more accurate rounding

        # update target policy action based on best value of actions
        action_space_ls = utils.get_action_space(episode[step][0])
        # take optimal action according to current state values
        optimal_action_idx = utils.get_optimal_action(
            Racetrack, episode[step][0], action_space_ls
        )
        Racetrack.target_policy_dict[x][y][v1][v2] = action_space_ls[optimal_action_idx]

        # compare if target_policy action matches current action taken
        if Racetrack.target_policy_dict[x][y][v1][v2] != episode[step][1]:
            logger.debug(
                "behaviour policy action and target policy action at step %d mismatch with current action value: %f vs expected action value: %f",
                step,
                Racetrack.get_state_value(next_state),
                Racetrack.get_state_value(utils.get_next_state(Racetrack, episode[step][0], Racetrack.target_policy_dict[x][y][v1][v2]))
            )
            return False

        # update importance sampling ratio
        w = w / ((1 - epsilon) + (epsilon / len(action_space_ls)))
        logger.debug("importance sampling weight W = %d at step %d", w, step)

    return True


def off_policy_control(
    Racetrack,
    epsilon=0.1,
    max_episode_count=10000,
    min_successful_episode=3,
    max_failed_episode_generation_attempt=1000,
    gamma=1,
):
    """
    Racetrack: Racetrack object from utils.
    max_episode_count: determines the maximum amount of successful episodes needed to be generated.
    min_successful_episode: determines the minimum amount of successful episodes generated for each starting positions.
    max_episode_generation_attempt: determines the maximum amount of any episode generated (success or failure).
    """
    # cumulative sum sampling ratio: state = 4d, action = 3d; total 7d
    cum_is = np.zeros(
        (len(Racetrack.racetrack), len(Racetrack.racetrack[0]), 5, 5, 3, 3), int
    )
    behaviour_policy = utils.get_policy(Racetrack, epsilon)
    episode_count = 0
    ep_generation_failed_attempt = 0
    successful_ep_counter = [0 for _ in range(len(Racetrack.start_coord_list))]
    successful_epi_dict = dict(zip(Racetrack.start_coord_list, successful_ep_counter))
    Episode = utils.Episode(Racetrack, behaviour_policy)
    logger.info("starting off-policy control")
    # tqdm.write("starting off-policy control")

    # define progress bar for epsilon value
    pbar_epsilon = tqdm(
        total=100,
        desc="current epsilon value for policy regeneration",
        position=0,
        leave=True,
        bar_format=pbar_format,
    )
    # start bar at initialized epsilon, scaled to 100 for better updates
    pbar_epsilon.n = int(epsilon * 100)
    pbar_epsilon.refresh()

    # define progress bar for successful episode count
    pbar_overall = tqdm(
        total=max_episode_count,
        desc="successful incremental prediction runs",
        position=1,
        leave=True,
        bar_format=pbar_format,
    )

    # define progress bars for successful episode's start state count, bars are a dict, so values are accessed from the key (state coord)
    pbar_starts = {
        coord: tqdm(
            total=min_successful_episode,
            desc=f"episode success for start coord {coord}",
            # mininterval=1,
            position=i + 2,  # +2 because the epsilon and overall take up position 0, 1
            leave=True,
            bar_format=pbar_format,
        )
        for i, coord in enumerate(Racetrack.start_coord_list)
    }

    while True:
        Episode.policy = behaviour_policy
        Episode.episode = []  # reset episode list
        if Episode.generate(Racetrack):
            episode_count += 1  # successful episode generated counter, seperated from the successful predictions
            if incremental_prediction(
                Racetrack, Episode.episode, cum_is, epsilon, gamma
            ):
                logger.debug("incremental prediction successful, episode count: %d", episode_count)
                pbar_overall.update(1)  # update for pbar too
                success_coord = (Episode.episode[-1][0][0], Episode.episode[-1][0][1])
                # counter for starting state for this successful episode
                successful_epi_dict[(success_coord)] += 1
                if pbar_starts[(success_coord)].n < min_successful_episode:
                    pbar_starts[(success_coord)].update(1)

            else:
                ep_generation_failed_attempt += 1  # general attempt counter
                logger.debug("incremental prediction failed, episode count: %d", episode_count)
                behaviour_policy = utils.get_policy(Racetrack, epsilon)
                continue

            if (min(successful_epi_dict.values()) >= min_successful_episode) and (
                episode_count >= max_episode_count
            ):
                logger.info(
                    "off-policy control achieved minimum successful episodes for each starting location, ending run..."
                )
                return True
            
            logger.debug(
                "Off-policy control successful at episode %d, with smallest success value: %d.",
                episode_count,
                min(successful_epi_dict.values())
            )
            behaviour_policy = utils.get_policy(Racetrack, epsilon)
            ep_generation_failed_attempt = 0

        else:
            ep_generation_failed_attempt += 1  # general attempt counter
            if ep_generation_failed_attempt > max_failed_episode_generation_attempt:
                logger.debug(
                    "episode generation attempts reached maximum of %d",
                    max_failed_episode_generation_attempt,
                )
                if epsilon < 1:
                    epsilon += 0.01
                    pbar_epsilon.update(1)
                else:
                    logger.info(
                        "epsilon value at maximum of 0.99, stopping off-policy control."
                    )
                    return False
                ep_generation_failed_attempt = 0
            else:
                logger.debug(
                    "episode generation reached max step, generation attempt %d failed...",
                    ep_generation_failed_attempt,
                )
                logger.debug("creating new behaviour policy with epsilon %.2f", epsilon)
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
        "####SSSSSS#########"
        ]

    race_track_obj = utils.Racetrack(race_track)

    MC_control_result = off_policy_control(
        Racetrack=race_track_obj,
        epsilon=0.1,
        max_episode_count=10000,
        min_successful_episode=10,
        max_failed_episode_generation_attempt=1000,
        gamma=0.9,
    )

    if MC_control_result == True:
        utils.generate_routes_gif(race_track_obj, race_track)
    else:
        print("Policy failed, ending algorithm.")


if __name__ == "__main__":
    main()
