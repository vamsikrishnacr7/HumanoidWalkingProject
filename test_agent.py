"""
test_agent.py - Test your trained agent

CREATE THIS FILE in the root HumanoidWalkingProject/ folder
(same level as test.py and quick_train.py)

Usage:
    python test_agent.py
"""

import numpy as np
import time
from pathlib import Path

from modules.env_humanoid import HumanoidWalkEnv
from modules.dqn_agent import DQNAgent


def main():
    print("="*70)
    print("TESTING TRAINED AGENT")
    print("="*70)
    
    # Try to find a trained model
    possible_models = [
        "models/pose_trained_best.pt",
        "models/quick_train_best.pt",
        "models/checkpoint_ep100.pt",
        "models/checkpoint_ep50.pt"
    ]
    
    model_path = None
    for path in possible_models:
        if Path(path).exists():
            model_path = path
            break
    
    if model_path is None:
        print(f"\n❌ No trained model found!")
        print("Please train the agent first by running:")
        print("  python quick_train.py")
        print("  OR")
        print("  python train_with_poses.py")
        return
    
    # Initialize environment
    env = HumanoidWalkEnv(urdf_path="data/humanoid.urdf", render=True)
    
    # Get state dimension
    dummy_obs = env.reset()
    state_dim = dummy_obs.shape[0]
    
    print(f"\nState dimension: {state_dim}")
    print(f"Loading model: {model_path}\n")
    
    # Initialize and load agent
    agent = DQNAgent(state_dim=state_dim, num_joints=4, actions_per_joint=5)
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during testing
    
    # Test for 3 episodes
    episodes = 3
    print(f"Running {episodes} test episodes...\n")
    
    for episode in range(episodes):
        print(f"{'='*70}")
        print(f"EPISODE {episode + 1}/{episodes}")
        print(f"{'='*70}")
        
        # Reset
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        max_forward_dist = 0
        
        # Run episode
        while not done and episode_length < 1000:
            # Select action (no exploration)
            action_indices, torques = agent.select_action(state, training=False)
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(torques)
            done = terminated or truncated
            
            # Track
            episode_reward += reward
            episode_length += 1
            
            # Track forward distance
            import pybullet as p
            pos, _ = p.getBasePositionAndOrientation(env.robot)
            max_forward_dist = max(max_forward_dist, pos[0])
            
            # Print every 100 steps
            if episode_length % 100 == 0:
                print(f"  Step {episode_length}: Reward={episode_reward:.1f}, "
                      f"Distance={max_forward_dist:.2f}m, Height={pos[2]:.2f}m")
            
            state = next_state
            time.sleep(1./240.)
        
        # Episode summary
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps: {episode_length}")
        print(f"  Max Distance: {max_forward_dist:.2f}m")
        
        if terminated:
            print(f"  Status: Terminated (fell)")
        else:
            print(f"  Status: Completed\n")
        
        # Pause between episodes
        if episode < episodes - 1:
            print("Next episode in 3 seconds...\n")
            time.sleep(3)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print("="*70)
    
    env.close()


if __name__ == "__main__":
    main()