import numpy as np
import racetrack_utils as utils
import logging
import logging.config

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
                "level": "INFO",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "racetrack.log",
                "formatter": "standard",
                "level": "DEBUG",
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],  # logs to both console and file
        },
        # this part only needed if want to customize logger in utils
        # "loggers": {    #   utils.__name__: {
        #       "level": "DEBUG"
        #   }
        # }
    }
)

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
        # update state value of current step using weighted importance sampling update rule
        Racetrack.state_values[x][y][v1][v2] = Racetrack.get_state_value(next_state) + (
            (w / (cum_is[x, y, v1, v2, a1, a2]))
            * (g - Racetrack.get_state_value(next_state))
        )

        # update target policy using greedy actions with respect to estimated values
        action_space_ls = utils.get_action_space(
            episode[step][0], Racetrack.racetrack[x][y]
        )
        optimal_action_idx = utils.get_optimal_action(
            Racetrack, episode[step][0], action_space_ls
        )
        Racetrack.target_policy_dict[x][y][v1][v2] = action_space_ls[optimal_action_idx]

        # compare if target_policy action matches current action taken
        if Racetrack.target_policy_dict[x][y][v1][v2] != episode[step][1]:
            logger.debug(
                "behaviour policy action and target policy action at step %d mismatch",
                step,
            )
            return False

        # update importance sampling ratio
        w = w / ((1 - epsilon) + (epsilon / len(action_space_ls)))
        logger.debug("importance sampling weight W = %d at step %d", w, step)


def off_policy_control(
    Racetrack,
    epsilon=0.1,
    max_episode_count=10000,
    min_successful_episode=3,
    max_episode_generation_attempt=10,
    gamma=0.9,
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
    ep_generation_attempt = 0
    successful_ep_counter = [0 for _ in range(len(Racetrack.start_coord_list))]
    successful_epi_dict = dict(zip(Racetrack.start_coord_list, successful_ep_counter))
    Episode = utils.Episode(Racetrack, behaviour_policy)
    while True:
        Episode.policy = behaviour_policy
        Episode.episode = []  # reset episode list
        if Episode.generate(Racetrack):
            if incremental_prediction(
                Racetrack, Episode.episode, cum_is, epsilon, gamma
            ):
                logger.debug("incremental prediction successful")
                episode_count += 1  # successful attempt counter
                ep_generation_attempt += 1  # general attempt counter
                successful_epi_dict[  # counter for starting state for this episode
                    (Episode.episode[-1][0][0], Episode.episode[-1][0][1])
                ] += 1
            else:
                logger.debug("creating new behaviour policy with epsilon %.2f", epsilon)
                behaviour_policy = utils.get_policy(Racetrack, epsilon)
                continue

            if (min(successful_epi_dict.values()) >= min_successful_episode) and (
                episode_count >= max_episode_count
            ):
                logger.debug(
                    "off-policy control achieved minimum successful episodes for each starting location, ending run..."
                )
                break
        else:
            ep_generation_attempt += 1  # general attempt counter
            if ep_generation_attempt > max_episode_generation_attempt:
                logger.debug(
                    "episode generation attempts reached maximum of %d",
                    max_episode_generation_attempt,
                )
                if epsilon < 1:
                    epsilon += 0.01
                else:
                    logger.info(
                        "epsilon value at maximum of 0.99, stopping off-policy control."
                    )
                    break
                ep_generation_attempt = 0
            else:
                logger.debug(
                    "episode generation reached max step, generation attempt %d failed...",
                    ep_generation_attempt,
                )
                logger.debug("creating new behaviour policy with epsilon %.2f", epsilon)
                behaviour_policy = utils.get_policy(Racetrack, epsilon)


def main():
    race_track = ["#######E", "#NNNNNNE", "#NNNNNNE", "#NNNNNNE", "#SS#####"]
    race_track_obj = utils.Racetrack(race_track)

    off_policy_control(
        Racetrack=race_track_obj,
        epsilon=0.1,
        max_episode_count=10000,
        min_successful_episode=4,
        max_episode_generation_attempt=20,
        gamma=0.9,
    )


main()
