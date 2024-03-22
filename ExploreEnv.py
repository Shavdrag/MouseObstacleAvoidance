import os

import numpy as np
import sys
from math import cos, sin, pi, sqrt, degrees
import gymnasium as gym
#from gym import spaces
from gym.utils import seeding

from Controller import MouseController
from ToSim import SimModel
from collections import deque
import random
import math




REWARD_DIAMETER = 0.15
REWARD_LOCATION = 1.5

def quat2euler(quat):
    # Function to convert quaternions to euler angles
    w, x, y, z = quat

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    X = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    Z = np.arctan2(t3, t4)

    return [X,Y,Z]
class CustomEnv(gym.Env):
    def __init__(self, modelPath, time_step=0.01, visual=False, num_states=5, num_frames=3, skip=10, map=0):

        super(CustomEnv, self).__init__()
        self.visual = visual
        self.time_step = time_step
        self.num_frames = num_frames
        self.skip = skip
        self.max_steps =  300*10    #400 for testing dynamics of whisker
        self.current_step = 0
        self.current_episode_reward = 0
        self.done = False
        self.upsidedown_threshold = np.radians(90)

        # Initialize simulation model
        self.theMouse = SimModel(modelPath, visual, time_step)
        self.theMouse.initializing()

        # Initialize controller
        spine = 20
        self.theController = MouseController(1, time_step, spine)
        self.theController.update_motion(max_fre=1, fre_g=1, spine_g=20, head_g=0)

        # Set up identifiers
        self.idx = self.theMouse.sim.model.body_name2id('mouse_head')
        self.whiskerids = [self.theMouse.sim.model.geom_name2id(name) for name in self.theMouse.sim.model.geom_names if
                           'whisk' in name]
        self.block_ids = self._get_block_ids()
        self.whisker_angles = np.array([0.0, 0.0, 0.0, 0.0])
        #self.whisker_history = np.array([0.0, 0.0, 0.0, 0.0])


        self.map = map
        self.grid_size = 30  # Define the grid size
        self.explored = False
        if self.map:

            self.world_bounds = (-2, 2)
            self.occupancy_grid = np.full((self.grid_size, self.grid_size), -1)
            #-1 unexplored, -0.5 free, 0.5 current pos, 1 obstacle



        # Define action space and observation space
        self.action_space = gym.spaces.Discrete(7)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_states*num_frames+self.map*self.grid_size*self.grid_size,))

        # Initialize location-related attributes
        self.aim_loc = [0, -REWARD_LOCATION]
        self.current_loc = self.theMouse.sim.data.get_site_xpos("body_ss")[:2]
        self.old_loc = np.copy(self.current_loc)

        #self.old_dist = self._calculate_distance(self.current_loc, self.aim_loc)
        self.last_action = 0.0
        self.euler = quat2euler(self.theMouse.sim.data.body_xquat[self.idx])

        # Precompute joint addresses for efficiency
        joint_names = ["whisker right front 12", "whisker right hind 12", "whisker left hind 12", "whisker left front 12"]
        self.joint_qpos_addrs = [self.theMouse.model.get_joint_qpos_addr(name) for name in joint_names]

        # Define action mappings
        self.action_mappings = {
            0: (1, 0), 1: (1, 1), 2: (1, -1),
            3: (-1, 0), 4: (-1, 1), 5: (-1, -1), 6: (0,0)
        }

        # Initialize frame stack

        self.frames = deque([], maxlen=num_frames)

        self.truncated = False

        self.delete = 0


    # MAP ----------------------------------------------------------------------
    def _discretize_position(self, position):
        x, y = position
        x_min, x_max = y_min, y_max = self.world_bounds
        grid_x = int((x - x_min) / (x_max - x_min) * self.grid_size)
        grid_y = int((y - y_min) / (y_max - y_min) * self.grid_size)
        return grid_x, grid_y

    def _update_occupancy_grid(self):
        # -1 unexplored, -0.5 free, 0.5 current pos, 1 obstacle

        old_position = self.old_loc
        new_position = self.current_loc
        collision_data = self._get_whisker_contact_points()

        old_x, old_y = self._discretize_position(old_position)
        new_x, new_y = self._discretize_position(new_position)

        if self.occupancy_grid[new_x, new_y] == -1:  # Assuming -1 for unexplored
            self.explored = True

        # Update the grid
        self.occupancy_grid[old_x, old_y] = 0  # Mark old position as free



        # Process collision points

        for collision_point in collision_data:
            col_x, col_y = self._discretize_position(collision_point)
            self.occupancy_grid[col_x, col_y] = 1  # Mark collision points as occupied


    # whisker --------------------------------------------------

    def _angle_to_grid_coords(self, angle, which_whisker):

        # Set direction based on  whisker (right: 0-1, left: 2-3)
        a = 1 if which_whisker > 1 else -1

        # Set root angle
        root_angle = 30 if which_whisker == 0 or which_whisker == 2 else 60

        WHISKER_LENGTH = 0.05  # Whisker length

        #  dx and dy based on the whisker angle and robot orientation
        dx = WHISKER_LENGTH * math.sin(math.radians(root_angle * a) + angle + self.euler[2])
        dy = WHISKER_LENGTH * math.cos(math.radians(root_angle) + angle + self.euler[2])

        # Calculate grid coordinates
        pos = self.theMouse.sim.data.body_xpos[self.idx][:2]
        x, y = pos[0] + dx, pos[1] + dy

        return (x, y)

    def _get_whisker_contact_points(self):
        # Get grid coordinates for whisker contact points
        contact_points = []
        for count, angle in enumerate(self.whisker_angles):
            if abs(angle) > 0.1:  # Threshold for detecting a contact
                contact_points.append(self._angle_to_grid_coords(angle, count))
                #print(angle)
        return contact_points




    # BASICSSS -----------------------------------------------------------------------
    def _get_block_ids(self):
        try:
            return [self.theMouse.sim.model.geom_name2id(name) for name in self.theMouse.sim.model.geom_names if
                    'block' in name]
        except Exception as e:
            print(f"Error getting block IDs: {e}")
            return None

    def _init_mouse(self):
        self.theController.update_motion(0,0,0,0,0)
        for _ in range(50*1):
            ctrlData = self.theController.runStep()
            self.theMouse.runStep(ctrlData, self.time_step)

        self.current_loc = self.theMouse.sim.data.get_site_xpos("body_ss")[:2]
        self.old_loc = np.copy(self.current_loc)
        self.theController.update_motion(max_fre=1, fre_g=1, spine_g=0, head_g=0)


    def step(self, action):
        #action = self.delete

        # Map the action to corresponding frequency and spine parameters
        fre_g, spine_g = self.action_mappings.get(action, (1, 0))

        if spine_g!=0 and action!=self.last_action:
            self.theController.update_motion(max_fre=1, fre_g=fre_g, spine_g=spine_g, head_g=0)
            for _ in range(10 * (self.skip + 1)):
                ctrlData = self.theController.runStep()
                self.theMouse.runStep(ctrlData, self.time_step)
        else:
            spine_step = spine_g/10
            for i in range(10):
                spine = (i+1)*spine_step
                self.theController.update_motion(max_fre=1, fre_g=fre_g, spine_g=spine, head_g=0)
                for _ in range((self.skip + 1)):
                    ctrlData = self.theController.runStep()
                    self.theMouse.runStep(ctrlData, self.time_step)


        self.euler = quat2euler(self.theMouse.sim.data.body_xquat[self.idx])
        self.last_action = action  # weil last action ja in abs
        obs = self.get_state()
        self.frames.append(obs)

        reward = self.calculate_reward(action)
        if reward>1:
            reward = -1
            self.truncated = True

        if self.map:
            try:
                self._update_occupancy_grid()
            except:
                self.truncated = True

        self.current_step += 1+self.skip
        self.current_episode_reward += reward

        if self.current_step >= self.max_steps:
            #print(self.current_episode_reward)
            
            self.truncated = True

        self.old_loc = np.copy(self.current_loc)


        self.frames.append(obs)

        #obs = self._get_stacked_obs()
        #print("Step observation shape:", obs.shape)
        return self._get_stacked_obs(), reward, self.done,self.truncated, {"location":self.current_loc, "orientation": quat2euler(self.theMouse.sim.data.body_xquat[self.idx])[2]}

    def calculate_reward(self, action):
        reward = -0.001


        if 1:#action<3:
            diff = np.linalg.norm(self.current_loc - self.old_loc)

            reward += diff
            if self.explored: #
                reward += 0.01
                self.explored = False

        if self.block_ids is not None and self.check_collision():
            reward = -0.1


        #if self.done: reward = 1

        if self.check_upsidedown():
            #print("upside down")
            self.done = True
            reward=-1

        return reward

    def check_collision(self):

        for i in range(self.theMouse.sim.data.ncon):
            contact = self.theMouse.sim.data.contact[i]
            if ((contact.geom1 in self.block_ids and self.theMouse.model.geom_bodyid[
                contact.geom2] not in self.whiskerids) or
                    (contact.geom2 in self.block_ids and self.theMouse.model.geom_bodyid[
                        contact.geom1] not in self.whiskerids)):
                return True

        if abs(self.whisker_angles[0]) > 0.5 or abs(self.whisker_angles[-1]) > 0.5:
            self.delete = 3
            return True

        return False


    def check_upsidedown(self):

        roll_angle, pitch_angle = self.euler[0:2]

        # Check roll or pitch angle indicates upside down
        if (np.abs(roll_angle - np.pi) < self.upsidedown_threshold or
                np.abs(pitch_angle - np.pi) < self.upsidedown_threshold):
            return True
        return False

    def reset(self, seed=None):
        super().reset()
        #self.action_space.seed(seed)
        print("%.2f" % self.current_episode_reward, end=" ")

        self.theMouse.sim.reset()
        self.theController.reset()
        self.current_step = 0
        self.done = False

        self.truncated=False


        if 0:  # Set to False to disable random spawning
            self._random_spawn()


        self._init_mouse()

        self.current_episode_reward = 0
        obs = self.get_state()

        self.old_loc = np.copy(self.current_loc)
        # self.set_hinge_joint_angle() also commented so fast manually switched

        # Reset frame stack
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(obs)

        #print("Reset observation shape:", self._get_stacked_obs().shape)
        return self._get_stacked_obs(), None

    def _random_spawn(self):
        body_id = self.theMouse.sim.model.body_name2id('mouse')
        addr = self.theMouse.sim.model.body_jntadr[body_id]

        # Generate random x and y positions
        random_x = np.random.uniform(-0.5, 0.5)
        random_y = np.random.uniform(-0.2, 0.2)

        # Update  position
        self.theMouse.sim.data.qpos[addr:addr + 3][:2] = [random_x, random_y]

    def get_state(self):

        # # Generate noise
        noise = np.random.normal(loc=0, scale=0.00, size=(4,))
        #print(noise)
        joint_positions = self.theMouse.sim.data.qpos[self.joint_qpos_addrs] + noise
        self.whisker_angles = joint_positions
        #self.whisker_history = np.concatenate((self.whisker_history,self.whisker_angles))


        self.current_loc = self.theMouse.sim.data.get_site_xpos("body_ss")[:2]



        state = np.hstack((joint_positions, self.last_action))#, self.euler[2], self.current_loc))

        return state

    def _get_stacked_obs(self):
        if len(self.frames) != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {len(self.frames)}")

        stacked_obs = np.hstack(self.frames)


        if self.map:
            flattened_grid = self.occupancy_grid.flatten()
            #print(stacked_obs)
            # Concatenate the flattened grid with the stacked frames
            stacked_obs = np.concatenate([stacked_obs, flattened_grid])

        return stacked_obs

    def set_hinge_joint_angle(self, joint_name="hinge_for_rotation"):
            # Convert the target angle to radians
            try:
                orient = random.randint(0,10)*10 - 50

                target_angle_radians = np.radians(orient)

                # Get index using  joint name
                joint_index = self.theMouse.sim.model.get_joint_qpos_addr(joint_name)
                # Set  target position for  joint
                self.theMouse.sim.data.qpos[joint_index] = target_angle_radians

                self.theMouse.sim.forward()
            except:
                pass

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        pass


    def close(self):
        # Clean up operation
        pass

