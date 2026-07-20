import gymnasium as gym
import gymnasium_env
from gymnasium.utils.env_checker import check_env

env = gym.make("gymnasium_env/WindyGridWorld-v0")

try:
    check_env(env.unwrapped)
    print("Environment passes all checks.")
except Exception as e:
    print(f"Environment has issues: {e}")