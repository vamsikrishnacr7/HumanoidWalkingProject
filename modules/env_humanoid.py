"""
modules/env_humanoid.py - Complete environment with DRL support

REPLACE your existing env_humanoid.py with this code.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os


class HumanoidWalkEnv:
    """Humanoid walking environment with reward function."""
    
    def __init__(self, urdf_path="data/humanoid.urdf", render=True):
        self.urdf_path = urdf_path
        self.render_mode = render
        
        # Connect to PyBullet
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(
            cameraDistance=2.5, 
            cameraYaw=50, 
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0.5]
        )
        
        # Joint control - hips unlocked with limited range for stability
        self.CONTROL_JOINTS = ['right_hip', 'right_knee', 'left_hip', 'left_knee']
        self.joint_indices = {}
        
        # Reward weights - TUNED FOR MOVEMENT
        self.w_vel = 5.0        # Forward velocity (MASSIVELY INCREASED!)
        self.w_live = 0.5       # Staying alive
        self.w_energy = 0.001  # Energy penalty (REDUCED)
        self.w_height = 0.3     # Height bonus
        
        # Termination
        self.min_height = 0.3
        self.max_angle = 0.7
        self.episode_step = 0
        self.max_episode_steps = 1000
        
        print("[INFO] HumanoidWalkEnv initialized")
    
    def reset(self, initial_pose=None):
        """Reset environment with STABLE initial pose."""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Load ground
        self.plane = p.loadURDF("plane.urdf")
        
        # Load robot
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        
        # Load at slightly higher position
        self.robot = p.loadURDF(self.urdf_path, [0, 0, 0.8], useFixedBase=False)
        
        # Build joint mapping
        self.joint_indices = {}
        for i in range(p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot, i)
            joint_name = joint_info[1].decode('utf-8')
            self.joint_indices[joint_name] = i
        
        # CRITICAL: Set joints to STABLE standing pose first
        if initial_pose is None:
            # Default stable standing pose
            stable_pose = {
                'right_hip': 0.0,      # Straight
                'right_knee': 0.0,     # Straight
                'right_ankle': 0.0,    # Flat
                'left_hip': 0.0,       # Straight
                'left_knee': 0.0,      # Straight
                'left_ankle': 0.0,     # Flat
            }
            
            # Apply stable pose
            for joint_name, angle in stable_pose.items():
                if joint_name in self.joint_indices:
                    joint_idx = self.joint_indices[joint_name]
                    p.resetJointState(self.robot, joint_idx, angle)
        else:
            # Apply custom initial pose
            for joint_name, angle in initial_pose.items():
                if joint_name in self.joint_indices:
                    joint_idx = self.joint_indices[joint_name]
                    p.resetJointState(self.robot, joint_idx, float(angle))
        
        # CRITICAL: Set base damping ONCE to stabilize upper body
        p.changeDynamics(
            self.robot, -1,  # -1 means base link (pelvis)
            linearDamping=2.5,      # Strong resistance to falling
            angularDamping=3.5      # Strong resistance to tipping
        )
        
        # LONGER stabilization with damping
        for _ in range(250):  # More steps for stability
            # Apply damping to ALL joints for stability
            for joint_name in self.joint_indices.keys():
                joint_idx = self.joint_indices[joint_name]
                
                # Different damping for different joint types
                if 'hip' in joint_name.lower():
                    force = 10.0  # High damping on hips for stability
                elif 'ankle' in joint_name.lower():
                    force = 8.0  # Moderate damping on ankles
                else:
                    force = 6.0  # Light damping on knees
                
                p.setJointMotorControl2(
                    self.robot, joint_idx,
                    p.VELOCITY_CONTROL,
                    targetVelocity=0,
                    force=force
                )
            p.stepSimulation()
        
        self.episode_step = 0
        obs = self.get_observation()
        return obs
    
    def get_observation(self):
        """Get state observation."""
        # Base state
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        vel, ang_vel = p.getBaseVelocity(self.robot)
        rpy = p.getEulerFromQuaternion(orn)
        
        base_state = np.array([
            pos[2], rpy[0], rpy[1],
            vel[0], vel[1], vel[2],
            ang_vel[0], ang_vel[1], ang_vel[2]
        ], dtype=np.float32)
        
        # Joint states
        joint_angles = []
        joint_velocities = []
        for i in range(p.getNumJoints(self.robot)):
            js = p.getJointState(self.robot, i)
            joint_angles.append(js[0])
            joint_velocities.append(js[1])
        
        return np.concatenate([
            base_state,
            np.array(joint_angles, dtype=np.float32),
            np.array(joint_velocities, dtype=np.float32)
        ])
    
    def step(self, action=None):
        """Execute one step."""
        # Apply action (torques) to leg joints
        if action is not None:
            action = np.clip(action, -0.8, 0.8)  # Reduce range slightly for stability
            max_torque = 40.0  # Reduced from 50 to prevent violent movements
            
            for i, joint_name in enumerate(self.CONTROL_JOINTS):
                if joint_name in self.joint_indices:
                    joint_idx = self.joint_indices[joint_name]
                    torque = action[i] * max_torque
                    p.setJointMotorControl2(
                        self.robot, joint_idx,
                        p.TORQUE_CONTROL, force=torque
                    )
        
        # CRITICAL: Keep ankles at 0° (flat feet) during training to prevent tip-over
        for joint_name in self.joint_indices.keys():
            if 'ankle' in joint_name.lower():
                joint_idx = self.joint_indices[joint_name]
                p.setJointMotorControl2(
                    self.robot, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=0.0,  # Flat feet
                    force=250.0,  # Strong stiffness to prevent foot rotation
                    positionGain=0.4
                )
        
        # Step simulation
        p.stepSimulation()
        self.episode_step += 1
        
        # Get new state
        obs = self.get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination
        terminated = self._check_termination()
        truncated = self.episode_step >= self.max_episode_steps
        
        info = {'episode_step': self.episode_step}
        
        # Render delay
        if self.render_mode:
            time.sleep(1.0 / 240.0)
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self, action):
        """Calculate reward - MAXIMIZED FOR FORWARD MOVEMENT."""
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        vel, ang_vel = p.getBaseVelocity(self.robot)
        
        # 1. Forward velocity reward (MAIN REWARD)
        r_vel = vel[0]  # x-axis velocity
        
        # 2. Alive bonus (staying upright)
        r_live = 1.0 if pos[2] > self.min_height else 0.0
        
        # 3. Height reward (encourage staying high)
        target_height = 0.6
        r_height = max(0, 1.0 - abs(pos[2] - target_height))
        
        # 4. Energy penalty (smooth movements) - VERY SMALL
        r_energy = np.sum(action ** 2) if action is not None else 0.0
        
        # Total reward
        reward = (self.w_vel * r_vel + 
                 self.w_live * r_live + 
                 self.w_height * r_height - 
                 self.w_energy * r_energy)
        
        # BIG penalty for falling
        if pos[2] < self.min_height:
            reward -= 30.0
        
        # ENORMOUS bonus for forward progress (ANY forward movement!)
        if r_vel > 0.01:  # Even tiny forward movement
            reward += 3.0
        if r_vel > 0.05:
            reward += 5.0
        if r_vel > 0.1:
            reward += 10.0  # Huge bonus for actual walking speed
        
        # Penalize backward movement strongly
        if r_vel < -0.01:
            reward -= 5.0
        
        return reward
    
    def _check_termination(self):
        """Check if episode should end."""
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        rpy = p.getEulerFromQuaternion(orn)
        
        # Fallen
        if pos[2] < self.min_height:
            return True
        
        # Tilted too much
        if abs(rpy[0]) > self.max_angle or abs(rpy[1]) > self.max_angle:
            return True
        
        return False
    
    def close(self):
        """Close environment."""
        p.disconnect()