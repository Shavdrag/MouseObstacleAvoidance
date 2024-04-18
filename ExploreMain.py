

from utility import *
# Built-in modules
import os
path = "/data/joris/mujoco210"

if os.path.exists(path): os.environ["MUJOCO_PY_MUJOCO_PATH"] = path
import gc
import sys
import cProfile
import pstats
# External libraries
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# Custom modules
from ExploreEnv import CustomEnv
from maze_generator import generate_obstacles, change_mazes
import os
import matplotlib.pyplot as plt
import math

DEFAULT_MODEL_PATH = "./models/dynamic_4l.xml" # Train in this maze if no obstacle
TIME_STEP = 0.005

wd = os.getcwd()
LOG_DIR = wd+"/log_dir/"

zip_dir = wd+"/trained_nets/"

NAME = "Whisker"#"WhiskerTest"

combined_name = zip_dir + NAME
LOAD_NAME = combined_name
SAVE_NAME = combined_name



WORKERS = 11      # amount of workers
VISUALIZE = False    # visualization (not recommended while training)
TRAIN = 1       # should the model train or just test
LR = 0.0003      # learning rate
obstacle = 1   # obstacles included (or not, basic walking)
N= 100            # Training cycles, how often trained and maze changed



def load_env(n_workers, time_step, visualize, modelPath, num_frames=10, frame_skip=9):
    """ Load environment """""

    if n_workers > 1:
        # for multi worker several environments
        nenvs = []
        n=0

        for i in range(n_workers):  # Different mazes for agents

            if obstacle: modelPath =  f"./models/dynamic_4l_maze_env{n}.xml"
            else: modelPath = DEFAULT_MODEL_PATH

            nenv = lambda: Monitor(CustomEnv(modelPath, time_step, visualize, num_states=5, num_frames=num_frames, skip=frame_skip), allow_early_resets=True)
            nenvs.append(nenv)

            n+=1
            if n>19: n=0

        env = SubprocVecEnv(nenvs)
        env = VecNormalize(env, norm_obs=0, norm_reward=1)

    else:
        env = DummyVecEnv([lambda: Monitor(CustomEnv(modelPath, time_step, visualize, num_states=5, num_frames=num_frames, skip=frame_skip))])
        env = VecNormalize(env, norm_obs=0, norm_reward=1)


    return env


def initialize_model(env, name, log_name):
    """Initialize or load a model."""
    
    if os.path.exists(name + '.zip'):
        model = PPO.load(name, env=env)
        print("log to   " + log_name)
        print("loaded model from   " + name)

    else:
        policy_kwargs = dict(net_arch=[256,256])
        model = PPO('MlpPolicy', env, n_steps=128 ,policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_name, learning_rate = LR)
        print("created new PPO model")

    print("   -    ")

    return model

def train_model(model, save_name):
    """Train the model."""
    n = 4000*10 # timesteps the model should train
    n_train = 1 # train several times so

    for i in range(n_train):
        a = random.uniform(0.9, 1.5)
        change_mazes(num_cubes=4, maze_area=[-a , a, -a, a], size=1 )

        model.learn(total_timesteps=n, reset_num_timesteps=False)

        model.save(save_name)
        print(i, f"/{n_train-1}   saved at {save_name}\n")


def test_model(model, env):
    """Test the trained model for vectorized environments."""

    obs = env.reset()
    num_envs = len(obs)

    actions = []
    angles = []
    reward_l = np.array([])

    all_pos = np.ones(shape=[1,2])
    contact_points = []
    done_flags = [False for _ in range(num_envs)]

    while not all(done_flags):
        action, _states = model.predict(obs)
        obs, rewards, done, info = model.env.step(action)
        done_flags = done

        #nobs = env.unnormalize_obs(obs)
        rewards = env.unnormalize_reward(rewards)

        if WORKERS==1 or VISUALIZE:
            #nobs = nobs.ravel()
            nobs = obs.ravel()

            test = nobs[:4]
            angles.append(test)  # Add angles for all workers
            actions.append(action)  # Add actioons for all workers

            position = info[0]["location"]
            orientation = info[0]["orientation"]

            all_pos = np.append(all_pos, [position], axis =0)

            reward_l = np.append(reward_l, rewards)

            new_points = get_whisker_contact_points(angles[-1], position, orientation)

            if new_points:
               contact_points.extend(new_points)
        else:
            angles.append(nobs[:, :4])
            actions.append([nobs[:, 4]])



    angles = np.delete(angles, (0), axis=0)
    actions = np.rint(actions).astype(int),

    return angles, reward_l, actions, contact_points, all_pos

def main(learn=True, visualize=False, n_workers=1):
    """Script for training and/or testing"""

    # Configuration and setup
    modelPath = DEFAULT_MODEL_PATH

    load_name = LOAD_NAME
    save_name = SAVE_NAME

    log_name = f"{LOG_DIR}{NAME}/"

    if visualize: n_workers = 1


    # ------ Training ------
    if learn:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        for i in range(N):

            # Generate random mazes
            a = random.uniform(0.9, 1.5)
            change_mazes(num_cubes=4, maze_area=[-a , a, -a, a], size=1 )

            env = load_env(n_workers=n_workers, time_step=TIME_STEP, visualize=visualize, modelPath=modelPath)


            model = initialize_model(env, load_name, log_name)

            # Train model (except needed when different amount of worker than in loaded model)
            try:
                train_model(model, save_name=save_name)
            except:
                policy_kwargs = dict(net_arch=[256, 256])
                new_model = PPO("MlpPolicy", env, n_steps=128,policy_kwargs=policy_kwargs,verbose=1, tensorboard_log=log_name)
                new_model.policy.load_state_dict(model.policy.state_dict(), strict=False)
                model = new_model
                del new_model
                train_model(model, save_name=save_name)

            # Close model and env
            env.close()
            del model
            gc.collect()
    

    # ------ Testing --------------
    env = load_env(n_workers=n_workers, time_step=TIME_STEP, visualize=visualize, modelPath=modelPath)
    model = PPO.load(load_name, env=env)

    all_actions = []

    angles, rewards, actions, contact_points, all_pos = test_model(model, env)
    all_actions.append(actions)

    # some possible visualization plots:
    # plot_action_hist(all_actions)   plot_collision_map(contact_points, all_pos, angles, WORKERS)  plot_angle_reward(angles, rewards[:-1])  plot_results(angles, rewards, actions)
    plot_angle_reward(angles, rewards[:-1])



if __name__ == "__main__":
    """Main function call"""

    main(learn=TRAIN, visualize=VISUALIZE, n_workers=WORKERS)

