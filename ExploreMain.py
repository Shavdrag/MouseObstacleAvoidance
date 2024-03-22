
"""
train model with new env (and new stif and damping)
curriculum learning env, fixed spot with reward behind o

TO DO:

    (include in env stay still option)

    maybe exclude tail from collision punishment

"""
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
from xml_whisker_generator import modify
import os
import matplotlib.pyplot as plt
import math
# Constants & Configuration

MODEL_PATH = "/data/joris/mujoco210"
#DEFAULT_MODEL_PATH = "./models/dynamic_4l_hingeorientation.xml"
DEFAULT_MODEL_PATH = "./models/dynamic_4l_maze_env11.xml"
TIME_STEP = 0.005 # * 10        *10        *3
#                     /step     /skip       /frames

wd = os.getcwd()
LOG_DIR = wd+"/log_dir/ "#"/../new_log_dir/"
LOG_DIR = wd+"/../new_log_dir/"

zip_dir = wd+"/trained_nets/"
#zip_dir = wd +"/../"

MAP = 0

#NAME =  "Walk" #V12 WhiskerOnlyV11_512"             #"3FWalkV4"

NAME = "Whisker"


LOAD_NAME = zip_dir + NAME#+"SOLO"
SAVE_NAME = zip_dir + NAME#+"SOLO"
#LOAD_NAME = SAVE_NAME
START_DEGREE = 40


STIFFNESS = 0.02     # 0.008
DAMPING = 0.007  # 0.002
# 0.01 0.01 nicthschlecht --
# 0.008, 0.001  48
SAVE = 1
TEST_ANGLES = 1     #max 10
WORKERS = 1
VISUALIZE = 1
TRAIN = 0
EXPLORE = 0.000
LR = 0.0003
obstacle = 1
# maybe add a torque actuator that only activates if no contact is there and set stiffness to very low
N= 100

#torch.set_num_threads(32)
#RANDOM_SEED = None
#torch.manual_seed(RANDOM_SEED)
 
#np.random.seed(RANDOM_SEED) 
#random.seed(RANDOM_SEED)

#PROBLEM:
#MAUS LÄUFT IMMER VOR UND ZURÜCK, SCHAUEN WIE DAS AUSSIEHT MIT MAP, vll exploitet sie das
# ->>>> Keep training the gripmap net after evaluation




def load_env(n_workers, time_step, visualize, modelPath, num_frames=10, frame_skip=9): # back to 10  at skip
    num_states = 5
    if n_workers > 1:
        nenvs = []
        n=0
        for i in range(n_workers):  # for mazes 0-20


            if obstacle: modelPath =  f"./models/dynamic_4l_maze_env{n}.xml"
            else: modelPath = DEFAULT_MODEL_PATH

            nenv = lambda: Monitor(CustomEnv(modelPath, time_step, visualize, num_states, num_frames, frame_skip), allow_early_resets=True)
            nenvs.append(nenv)

            n+=1
            if n>19: n=0

        env = SubprocVecEnv(nenvs)
        env = VecNormalize(env, norm_obs=0, norm_reward=1)

    else:
        env = DummyVecEnv([lambda: Monitor(CustomEnv(modelPath, time_step, visualize, num_states, num_frames, frame_skip))])
        env = VecNormalize(env, norm_obs=1, norm_reward=1)


    return env


def initialize_model(env, name, log_name):
    """Initialize or load a model."""
    if os.path.exists(name + '.zip'):
        model = PPO.load(name, env=env)
        model.ent_coef = EXPLORE         #WIEDER AN WENN PPO
        model.learning_rate = LR
        model.tensorboard_log=log_name
        print("log to   "+log_name)
        print("loaded model from   " + name)
    else:

        policy_kwargs = dict(net_arch=[256,256])
        model = PPO('MlpPolicy', env, n_steps=128 ,policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_name, learning_rate = LR)
        #model.ent_coef = EXPLORE         WIEDER AN WENN PPO

        print("created new PPO model")

    print("   -    ")
    #print(model.policy)
    print("   -    ")
    if 0:
        print("Model parameters:")
        for key, value in model.policy.__dict__.items():
            if key[0] != '_':  # This will filter out private attributes
                print(f"{key}: {value}")

    return model

def train_model(model, env,n_train, save_name, save):
    """Train the model."""
    n = 4000*10 #2048*10
    n_train = 1
    for i in range(n_train):
        a =random.uniform(0.9, 1.5)
        change_mazes(num_cubes=4, maze_area=[-a , a, -a, a], size=1 )

        model.learn(total_timesteps=n, reset_num_timesteps=False)

        # Evaluate the policy and log the results
        #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=4)
        #model.logger.record('mean_reward', mean_reward)
        #model.logger.dump(step=n)

        if save:
            model.save(save_name)
            print(i, f"/{n_train-1}   saved at {save_name}\n")


def test_model(model, env):
    """Test the trained model for vectorized environments."""


    #    PROBLEM IRGENDWO AN UNTERSCHIED IN DER KLASSE SELBST UND DANN VBEIM TESTEN
    obs = env.reset()
    num_envs = len(obs)
    print("num envs:   ", num_envs)


    actions = []
    angles = []

    reward_l = np.array([])    #[np.array([]) for _ in range(num_envs)]

    done_flags = [False for _ in range(num_envs)]
    #print(done_flags)
    all_pos = np.ones(shape=[1,2])

    contact_points = []
    while not all(done_flags):
        action, _states = model.predict(obs)
        obs, rewards, done, info = model.env.step(action)
        #print("Observations in script :   ", obs)
        done_flags = done


        #print(done)
        #sys.exit()


        if 1:

            # jointangles 1-4 , action , orientation, x, y
            #nobs = obs.ravel()
            nobs = env.unnormalize_obs(obs)
            rewards = env.unnormalize_reward(rewards)

            if WORKERS==1 or VISUALIZE:
                nobs = nobs.ravel()

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
                angles.append(nobs[:, :4])  #
                actions.append([nobs[:, 4]])



    angles = np.delete(angles, (0), axis=0)
    #print(actions)
    actions = np.rint(actions).astype(int)
    #print(actions)

    return angles, reward_l, actions, contact_points, all_pos

def main(learn=True, visualize=False, n_workers=1):
    # Configuration and setup
    modelPath = DEFAULT_MODEL_PATH

    load_name = LOAD_NAME
    save_name = SAVE_NAME

    log_name = f"{LOG_DIR}{NAME}/"


    #if visualize: n_workers = 1

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if learn:
        for i in range(N):
            a =random.uniform(0.9, 1.5)
            change_mazes(num_cubes=4, maze_area=[-a , a, -a, a], size=1 )
            env = load_env(n_workers=n_workers, time_step=TIME_STEP, visualize=visualize, modelPath=modelPath)
            # env1 = load_env(n_workers=n_workers,time_step=TIME_STEP,visualize=visualize, modelPath=modelPath)


            if 1:
                if WORKERS==1:
                    #normal = "solono.pkl"
                    #env = VecNormalize.load(normal, env)
                    pass
                else:
                    normal = "last.pkl"
                    env = VecNormalize.load(normal, env)

            n_train = 1

            model = initialize_model(env, load_name, log_name)
            try:
                train_model(model, n_train=n_train,env=env, save_name=save_name, save=SAVE)
            except:

                policy_kwargs = dict(net_arch=[256, 256])
                new_model = PPO("MlpPolicy", env, n_steps=128,policy_kwargs=policy_kwargs,verbose=1, tensorboard_log=log_name, ent_coef=EXPLORE)
                new_model.policy.load_state_dict(model.policy.state_dict(), strict=False)
                model = new_model
                del new_model
                train_model(model, n_train=n_train,env=env, save_name=save_name, save=SAVE)

            env.save(normal)
            env.close()
            del model
            gc.collect()
    
    all_angles = []
    all_actions = []
    #sys.exit()

    env = load_env(n_workers=n_workers, time_step=TIME_STEP, visualize=visualize, modelPath=modelPath)

    # Load model outside  loop
    #env = VecNormalize.load("solo.pkl", env) #normal.pkl oder solo




    model = PPO.load(load_name, env=env)

    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    #print(f"Mean reward: {mean_reward} +/- {std_reward}")
    
    #sys.exit()

    for i in range(1):
        angles, rewards, actions, contact_points, all_pos = test_model(model, env)
        all_actions.append(actions)

    #plot_action_hist(all_actions)
    plot_collision_map(contact_points, all_pos, angles, WORKERS)
    #plot_angle_reward(angles, rewards)#

    #plot_results(angles, rewards, actions)




    threshold = 0.005
    #  boolean mask
    mask = angles > threshold

    # element  over the threshold
    rows_with_condition = np.any(mask, axis=1)


    if rows_with_condition.any():
        #
        index_reversed_rows = np.argmax(rows_with_condition[::-1])
        row_index = len(angles) - 1 - index_reversed_rows
    else:

        row_index = 80
    return -row_index
    #print("Row index:", row_index, "/", angles.shape[0])


    if 0:
        # Ensure the directory exists
        save_dir = "orientationStudy"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the angles and actions
        np.save(os.path.join(save_dir, "all_angles.npy"), all_angles_array)
        np.save(os.path.join(save_dir, "all_actions.npy"), all_actions_array)



if __name__ == "__main__":
    search = 10
    stiffness_range = np.linspace(0.0001, 0.02, search)
    damping_range = np.linspace(0.0001, 0.02, search)

    top_scores = []


    vis = VISUALIZE
    num = WORKERS
    learn = TRAIN

    # Main function call
    main(learn=learn, visualize=vis, n_workers=num)

    if 0:
        count=0
        # Grid search
        for stiffness in stiffness_range:
            for damping in damping_range:
                count +=1
                modify(stiffness, damping)
                # Assuming 'learn', 'vis', and 'num' are predefined variables
                print("now ", count, "/",search*search)
                score = main(learn=learn, visualize=vis, n_workers=num)

                # Add score and parameters to the list
                top_scores.append((score, stiffness, damping))

        # Sort the list by score in descending order and take the top 5
        top_scores.sort(reverse=True, key=lambda x: x[0])
        top_5_scores = top_scores[:5]

        # Print the top 5 parameter combinations and their scores
        print("Top 5 Parameter Combinations:")
        for rank, (score, stiffness, damping) in enumerate(top_5_scores, start=1):
            print(f"{rank}. Score: {score}, STIFFNESS: {stiffness}, DAMPING: {damping}")

"""

-1.27 Top 5 whisker Parameter Combinations:
1. Score: -22, STIFFNESS: 0.01778888888888889, DAMPING: 0.006733333333333333
2. Score: -22, STIFFNESS: 0.02, DAMPING: 0.006733333333333333
3. Score: -24, STIFFNESS: 0.01778888888888889, DAMPING: 0.004522222222222223
4. Score: -24, STIFFNESS: 0.02, DAMPING: 0.004522222222222223
5. Score: -25, STIFFNESS: 0.013366666666666666, DAMPING: 0.002311111111111111
"""

