import random
import pprint
import numpy as np
import gymnasium as gym
import gymnasium_env
from collections import defaultdict

# For visualisation and save location usage
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
from PIL import Image


# Base random seed
BASE_RANDOM_SEED = 123456789


def TD_Sarsa(
    env,
    episodes=1000,
    seed=BASE_RANDOM_SEED,
    alpha=0.5,  # delta learning-rate
    gamma=1,  # Next state-action pair weighing/ discounting-factor
    epsilon=0.1,  # greedy-epsilon policy
    render=None,
    render_path="",
    stochastic=False,
    run_number=0
):
    np.random.seed(seed)
    random.seed(seed)

    # Designate output gif directory
    gifs_dir = Path(__file__).resolve().parent / "windy_gridworld_gifs"
    gifs_dir.mkdir(exist_ok=True)
    output_dir = Path(gifs_dir / f"windy_{run_number}")
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True)

    # List for gif images
    images = []

    env.render_mode = render
    env.render_path = render_path
    env.stochastic = stochastic

    n_states_x = env.x_size
    n_states_y = env.y_size
    n_actions = env.action_space.n
    q_table = np.zeros(shape=(n_states_y, n_states_x, n_actions))

    # episode_dict is used to keep track of amount of steps taken for each episode.
    episodes_dict = defaultdict(int)
    episodes_dict["0"] = 0
    episode_num = 1

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
            next_state, reward, done, truncated, info = env.step(action, episode + 1)
            total_reward += reward

            # Sarsa update
            if not (done or truncated):
                if np.random.random() < epsilon:
                    next_q = q_table[next_state["agent"][0], next_state["agent"][1]][
                        np.random.randint(0, n_actions)
                    ]
                else:
                    next_q = np.max(
                        q_table[next_state["agent"][0], next_state["agent"][1]]
                    )
            else:
                # q value for terminal step
                next_q = 0

            q_table[state["agent"][0], state["agent"][1], action] += alpha * (
                reward
                + (gamma * next_q)
                - q_table[state["agent"][0], state["agent"][1], action]
            )

            state = next_state
            episodes_dict[episode_num] += 1

            # Creates gif, NOTE that render has to be either "human" or "rgb_array"
            if render != None:
                if episode == episodes - 1:
                    shutil.copy(Path(__file__).resolve().parent / "final_episode.png", output_dir / f"Final episode step {episodes_dict[episode_num]}.png")
                    image = Image.open(output_dir / f"Final episode step {episodes_dict[episode_num]}.png")
                    images.append(image)
        
        episode_num += 1

    if render != None:
        images[0].save(
                output_dir
                / f"Optimal_path_for_{run_number}.gif",
                save_all=True,
                append_images=images[1:],
                duration=200,
                loop=0,
            )

    return episodes_dict


def main():
    n_runs = 10
    episodes = 1000
    alpha = 0.5  # delta learning-rate
    gamma = 1  # Next state-action pair weighing/ discounting-factor
    epsilon = 0.1  # greedy-epsilon policy
    source_path = Path(__file__).resolve().parent
    image_path = source_path / "run_graphs"
    image_path.mkdir(exist_ok=True)

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
            env,
            episodes=episodes,
            seed=seed,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            render="rgb_array",
            render_path=source_path,
            stochastic=False,
            run_number=i + 1,
        )
        results_list.append(results)
        # Extract keys and values
        y = list(results.keys())
        x = list(results.values())
        x_cumsum = np.cumsum(x)

        plt.figure()
        plt.plot(x_cumsum, y, "b-")
        plt.xlabel("Steps Taken")
        plt.ylabel("Episodes")
        plt.xlim(0, round(x_cumsum[170], -3))
        plt.ylim(-5, 170)
        plt.yticks([0, 50, 100, 150, 170])
        plt.title("Steps taken for each episodes")
        plt.savefig(image_path / f"run_{i + 1:04d}.png")


if __name__ == "__main__":
    main()
