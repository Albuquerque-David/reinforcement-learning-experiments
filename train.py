import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
import os
import time

import stable_baselines3

models_dir = f"models/donkeykong/DQN-{int(time.time())}"
logdir = f"logs/donkeykong/DQN-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("ALE/DonkeyKong-v5", render_mode="human")
env.reset()

print(stable_baselines3.common.utils.get_device(device='auto'))

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    




# episodes = 10
# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         env.render()
#         observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
#         # print(reward)

env.close()