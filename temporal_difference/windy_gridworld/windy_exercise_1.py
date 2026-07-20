import gymnasium as gym
import gymnasium_env
from gymnasium.utils.env_checker import check_env

# Note: gymnasium_env was imported as a local lib to be detected by script to check custom envs, else it would raise an Namespace error in the gymnasium lib

env = gym.make("gymnasium_env/WindyGridWorld-v0")

try:
    check_env(env.unwrapped)
    print("Environment passes all checks.")
except Exception as e:
    print(f"Environment has issues: {e}")