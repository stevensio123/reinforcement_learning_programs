import random
import pprint
import numpy as np
import gymnasium as gym
import gymnasium_env
from collections import defaultdict

# For visualisation and save location usage
from pathlib import Path
import matplotlib.pyplot as plt


# Base random seed
BASE_RANDOM_SEED = 123456789


def TD_Sarsa(
    env,
    episodes=1000,
    seed=BASE_RANDOM_SEED,
    alpha=0.5,  # delta learning-rate
    gamma=1,  # Next state-action pair weighing/ discounting-factor
    epsilon=0.1,  # greedy-epsilon policy
):
    np.random.seed(seed)
    random.seed(seed)

    n_states_x = env.x_size
    n_states_y = env.y_size
    n_actions = env.action_space.n
    q_table = np.zeros(shape=(n_states_y, n_states_x, n_actions))

    # For reward tracking
    episodes_dict = defaultdict(int)
    episode_num = 0

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # For discovery/exploration action
            if np.random.random() < epsilon:
                action = np.random.randint(0, n_actions)

            # For greedy action
            else:
                action = np.argmax(q_table[state["agent"][0], state["agent"][1]])

            # Take action and observe result
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Sarsa update
            if not (done or truncated):
                next_max = np.max(
                    q_table[next_state["agent"][0], next_state["agent"][1]]
                )
                q_table[state["agent"][0], state["agent"][1], action] += alpha * (
                    reward
                    + (gamma * next_max)
                    - q_table[state["agent"][0], state["agent"][1], action]
                )

            state = next_state
            episodes_dict[episode_num] += 1
        episode_num += 1
    return episodes_dict


# return {"episodes taken": episodes}


def main():
    n_runs = 10
    episodes = 1000
    alpha = 0.5  # delta learning-rate
    gamma = 1  # Next state-action pair weighing/ discounting-factor
    epsilon = 0.1  # greedy-epsilon policy

    # Generate different seeds for each run
    seeds = [BASE_RANDOM_SEED + i for i in range(n_runs)]
    # Store results for comparison
    results_list = []

    # Run experiments with different seeds
    for i, seed in enumerate(seeds):
        print(f"Run {i + 1}/{n_runs} with seed {seed}")
        env_wrapped = gym.make("gymnasium_env/WindyGridWorld-v0")
        env = env_wrapped.unwrapped
        results = TD_Sarsa(
            env, episodes=episodes, seed=seed, alpha=alpha, gamma=gamma, epsilon=epsilon
        )
        results_list.append(results)
        # Extract keys and values
        x = list(results.keys())
        y = list(results.values())
        # y_cumsum = np.cumsum(y)

        plt.figure()
        plt.plot(x, y, "b-")
        plt.xlabel("Episode Number")
        plt.ylabel("Steps Taken")
        plt.title("Steps taken for each episodes")
        plt.savefig(f"run_{i:04d}.png")
        # plt.show()

        # pprint.pprint(results, indent=4, width=60, sort_dicts=True)
    # print(f"all runs completed, all results:{results_list}")


if __name__ == "__main__":
    main()
