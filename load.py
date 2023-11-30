import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
import os
import time


env = gym.make("ALE/DonkeyKong-v5", render_mode="human")
env.reset()

models_dir = "models/donkeykong/PPO-1700512027"
model_path = f"{models_dir}/60000.zip"

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