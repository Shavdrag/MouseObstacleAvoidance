import os

import numpy as np
import sys
from math import cos, sin, pi, sqrt, degrees
import gymnasium as gym
from gym.utils import seeding

from Controller import MouseController
from ToSim import SimModel
from collections import deque
import random
import math


def quat2euler(quat):
    """Convert quaternions to euler angles (UTILITY FUNCTION)"""

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

    def __init__(self, modelPath, time_step=0.01, visual=False, num_states=5, num_frames=3, skip=10):
        """Initialize parameters of environment and precompute variables"""

        super(CustomEnv, self).__init__()
        self.visual = visual
        self.time_step = time_step
        self.num_frames = num_frames
        self.skip = skip

        self.max_steps = 300*10
        self.current_step = 0
        self.current_episode_reward = 0
        self.done = False
        self.upsidedown_threshold = np.radians(90)

        # Initialize simulation model and controller
        self.theMouse = SimModel(modelPath, visual, time_step)
        self.theMouse.initializing()

        spine = 20
        self.theController = MouseController(1, time_step, spine)
        self.theController.update_motion(max_fre=1, fre_g=1, spine_g=20, head_g=0)

        # Set up identifiers
        self.idx = self.theMouse.sim.model.body_name2id('mouse_head')
        self.whiskerids = [self.theMouse.sim.model.geom_name2id(name) for name in self.theMouse.sim.model.geom_names if
                           'whisk' in name]
        self.block_ids = [self.theMouse.sim.model.geom_name2id(name) for name in self.theMouse.sim.model.geom_names if
                    'block' in name]
        self.whisker_angles = np.array([0.0, 0.0, 0.0, 0.0])

        # Define action space and observation space
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_states*num_frames,))

        # Initialize location-related attributes
        self.current_loc = self.theMouse.sim.data.get_site_xpos("body_ss")[:2]
        self.old_loc = np.copy(self.current_loc)

        # Set for first observation
        self.last_action = 0.0
        self.euler = quat2euler(self.theMouse.sim.data.body_xquat[self.idx])

        # Precompute whisker joint addresses for efficiency
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


    def _init_mouse(self):
        """Set motion and init simulation"""

        self.theController.update_motion(0,0,0,0,0)
        for _ in range(50*1):
            ctrlData = self.theController.runStep()
            self.theMouse.runStep(ctrlData, self.time_step)

        self.current_loc = self.theMouse.sim.data.get_site_xpos("body_ss")[:2]
        self.old_loc = np.copy(self.current_loc)
        self.theController.update_motion(max_fre=1, fre_g=1, spine_g=0, head_g=0)


    def step(self, action):
        """Proceed in simulation with given action"""

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
        self.last_action = action
        obs = self.get_state()
        self.frames.append(obs)

        reward = self.calculate_reward()

        if reward>1:
            # Sometimes mouse glitches, then mouse get catapulted - punish this and stop episode
            reward = -1
            self.truncated = True

        self.current_step += 1+self.skip
        self.current_episode_reward += reward

        if self.current_step >= self.max_steps:
            self.truncated = True

        self.old_loc = np.copy(self.current_loc)

        self.frames.append(obs)

        return self._get_stacked_obs(), reward, self.done,self.truncated, {"location":self.current_loc, "orientation": quat2euler(self.theMouse.sim.data.body_xquat[self.idx])[2]}


    def calculate_reward(self):
        """Calculate reward as feedback"""

        diff = np.linalg.norm(self.current_loc - self.old_loc)

        reward = diff

        if self.block_ids is not None and self.check_collision():
            reward = -0.1

        if self.check_upsidedown():
            self.done = True
            reward=-1

        return reward


    def check_collision(self):
        """Check if mouse collides with blocks (except whiskers)"""

        for i in range(self.theMouse.sim.data.ncon):
            contact = self.theMouse.sim.data.contact[i]
            if ((contact.geom1 in self.block_ids and self.theMouse.model.geom_bodyid[
                contact.geom2] not in self.whiskerids) or
                    (contact.geom2 in self.block_ids and self.theMouse.model.geom_bodyid[
                        contact.geom1] not in self.whiskerids)):
                return True

        if abs(self.whisker_angles[0]) > 0.5 or abs(self.whisker_angles[-1]) > 0.5:
            return True

        return False


    def check_upsidedown(self):
        """Check roll or pitch angle indicates upside down (mouse laying on back)"""

        roll_angle, pitch_angle = self.euler[0:2]

        if (np.abs(roll_angle - np.pi) < self.upsidedown_threshold or
                np.abs(pitch_angle - np.pi) < self.upsidedown_threshold):
            return True
        return False


    def reset(self, seed=None):
        """Reset the environment"""

        super().reset()
        print("%.2f" % self.current_episode_reward, end=" ")

        self.theMouse.sim.reset()
        self.theController.reset()
        self.current_step = 0
        self.done = False

        self.truncated=False

        self._init_mouse()

        self.current_episode_reward = 0
        obs = self.get_state()

        self.old_loc = np.copy(self.current_loc)

        # Reset frame stack
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(obs)

        return self._get_stacked_obs(), None


    def get_state(self):
        """Get observations"""

        joint_positions = self.theMouse.sim.data.qpos[self.joint_qpos_addrs]
        self.whisker_angles = joint_positions

        self.current_loc = self.theMouse.sim.data.get_site_xpos("body_ss")[:2]

        state = np.hstack((joint_positions, self.last_action))

        return state


    def _get_stacked_obs(self):
        """Convert frames to observation (frame stacking)"""

        if len(self.frames) != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {len(self.frames)}")

        stacked_obs = np.hstack(self.frames)

        return stacked_obs




    # FUNCTIONS FOR GYM-METHODOLOGY ONLY ----------------------------------------------------------------------------

    def render(self, mode="human"):
        pass

    def seed(self, seed=None):
        pass

    def close(self):
        pass





    # FOR FIGURES ONLY, NOT IMPORTANT FOR FUNCTIONALITY  -------------------------------------------------------------

    def _angle_to_grid_coords(self, angle, which_whisker):
        # FOR FIGURES ONLY
        """Computes rough location of whisker collision point"""

        # Set direction based on  whisker (right: 0-1, left: 2-3)
        a = 1 if which_whisker > 1 else -1

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
        # FOR FIGURES ONLY
        """Get grid coordinates for whisker contact points"""

        contact_points = []
        for count, angle in enumerate(self.whisker_angles):
            if abs(angle) > 0.1:  # Threshold for detecting a contact
                contact_points.append(self._angle_to_grid_coords(angle, count))

        return contact_points
