import random
import numpy as np
import gymnasium as gym

# For visualisation and save location usage
from pathlib import Path
import matplotlib.pyplot as plt


# Base random seed
BASE_RANDOM_SEED = 123456789

def TD_Sarsa(
        env,
        episodes=1000,
        seed=BASE_RANDOM_SEED,
        alpha=0.5, # delta learning-rate
        gamma=1, # Next state-action pair weighing/ discounting-factor
        epsilon=0.1, # greedy-epsilon policy
):
    np.random.seed(seed)
    random.seed(seed)

    n_states_x = env.x_size
    n_states_y = env.y_size
    n_actions = env.action_space.n
    q_table = np.zeros(shape=(n_states_y, n_states_x, n_actions))

    # For reward tracking
    step_tracker = 0

    for episode in range(episodes):
        state, info = env.reset(seed=seed+episode)
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):

            # For discovery/exploration action
            if np.random.random() < epsilon:
                action = np.random.randint(0, n_actions)

            # For greedy action
            else:
                action = np.argmax(q_table[state["agent"][0],state["agent"][1]])

            # Take action and observe result
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Sarsa update
            if not (done or truncated):
                next_max = np.max(q_table[next_state["agent"][0], next_state["agent"][1]])
                q_table[state["agent"][0], state["agent"][1], action] += alpha * (reward + (gamma * next_max) - q_table[state["agent"][0], state["agent"][1], action])

            state = next_state

        step_tracker += 1

    return {
        "steps_taken": step_tracker
    }