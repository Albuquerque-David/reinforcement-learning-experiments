import gymnasium as gym

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack
from gymnasium.wrappers import FrameStack
from stable_baselines3.common.vec_env import VecTransposeImage
from datetime import datetime

import os
import time
import torch

timestamp = datetime.utcfromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S')
models_dir = f"models/breakout/PPO-{timestamp}"
logdir = f"logs/breakout/PPO-{timestamp}"

device = torch.device('cuda')
torch.set_default_device(device)
print(device)


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")

print(stable_baselines3.common.utils.get_device(device='cuda'))

# n_envs=8
# n_steps=128 * 8, n_epochs=4, batch_size=256, learning_rate=0.00025, clip_range=0.1, vf_coef=0.5, ent_coef=0.01
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=logdir, n_steps=128, n_epochs=4, batch_size=256, learning_rate=0.00025, clip_range=0.1, vf_coef=0.5, ent_coef=0.01)
# model = PPO("MlpPolicy", env, verbose=1)


TIMESTEPS = 100000
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

env.close()