import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
import os
import time


env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
env.reset()

models_dir = "models/breakout/PPO-2024-01-08 00:43:35"
model_path = f"{models_dir}/5900000.zip"

model = PPO.load(model_path, env=env)

vec_env = model.get_env()
episodes = 10
for ep in range(episodes):
    obs = vec_env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, terminated, info = vec_env.step(action)
        # print(reward)

env.close()