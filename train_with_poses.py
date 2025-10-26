"""
train_with_poses.py - Train with initial poses from images

CREATE THIS NEW FILE in the root folder.

This version uses your pose detection modules to initialize the humanoid
with different poses from your images.

Usage:
    python train_with_poses.py
"""

import numpy as np
from pathlib import Path
import random

from modules.env_humanoid import HumanoidWalkEnv
from modules.dqn_agent import DQNAgent

# Import your pose detection modules
try:
    from modules.module1_pose import run_on_image
    from modules.module1_to_angles import compute_full_body_angles
    from modules.main_character_selector import select_main_character
    from modules.pose_to_pybullet import AngleToPyBulletMapper, validate_pose
    POSE_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import pose modules: {e}")
    POSE_DETECTION_AVAILABLE = False


def load_pose_library(image_folder="poses/sample_images", max_poses=20):
    """Load poses from images in the folder."""
    if not POSE_DETECTION_AVAILABLE:
        print("Pose detection not available. Training with default poses only.")
        return []
    
    print(f"\n{'='*70}")
    print("LOADING POSE LIBRARY FROM IMAGES")
    print(f"{'='*70}")
    
    # Get image files
    image_folder = Path(image_folder)
    if not image_folder.exists():
        print(f"Warning: Folder {image_folder} not found!")
        return []
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_folder.glob(f"*{ext}")))
        image_files.extend(list(image_folder.glob(f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("No images found!")
        return []
    
    # Process images
    pose_library = []
    mapper = AngleToPyBulletMapper(view_type='front')
    
    for i, img_path in enumerate(image_files[:max_poses]):
        try:
            print(f"Processing {img_path.name}...", end=" ")
            
            # Detect pose
            vis_img, people_data = run_on_image(str(img_path))
            
            if len(people_data) == 0:
                print("❌ No person detected")
                continue
            
            # Select main character
            main_idx, main_skeleton, selection_info = select_main_character(
                people_data, method='weighted'
            )
            
            # Compute angles
            detected_angles = compute_full_body_angles(main_skeleton)
            
            # Convert to PyBullet format
            pb_angles = mapper.convert_angles(detected_angles, debug=False)
            
            # Validate
            is_valid, warnings = validate_pose(pb_angles)
            
            if is_valid:
                pose_library.append(pb_angles)
                print(f"✓ Pose {len(pose_library)} added")
            else:
                print(f"⚠️  Invalid pose")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print(f"\n✓ Loaded {len(pose_library)} valid poses")
    print(f"{'='*70}\n")
    
    return pose_library


def main():
    print("="*70)
    print("TRAINING WITH IMAGE-BASED INITIAL POSES")
    print("="*70)
    print("\nThis will train the humanoid to walk from various starting poses")
    print("extracted from your images.\n")
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    
    # Load pose library
    pose_library = load_pose_library(image_folder="poses/sample_images", max_poses=20)
    
    if len(pose_library) > 0:
        print(f"Training will use {len(pose_library)} different starting poses!")
    else:
        print("No poses loaded. Training with default standing pose.")
    
    # Initialize environment
    env = HumanoidWalkEnv(urdf_path="data/humanoid.urdf", render=True)
    
    # Get state dimension
    dummy_obs = env.reset()
    state_dim = dummy_obs.shape[0]
    
    print(f"\nState dimension: {state_dim}")
    print(f"Control joints: {len(env.CONTROL_JOINTS)}\n")
    
    # Initialize agent
    agent = DQNAgent(state_dim=state_dim, num_joints=4, actions_per_joint=5)
    
    # Training
    episodes = 100
    print(f"Starting training for {episodes} episodes...\n")
    
    episode_rewards = []
    best_reward = -999
    
    for episode in range(episodes):
        # Select random initial pose (if available)
        if len(pose_library) > 0:
            initial_pose = random.choice(pose_library)
            print(f"Ep {episode + 1}: Using random pose from library")
        else:
            initial_pose = None
        
        # Reset environment with pose
        state = env.reset(initial_pose=initial_pose)
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Run episode
        while not done and episode_length < 500:
            # Select action
            action_indices, torques = agent.select_action(state, training=True)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(torques)
            done = terminated or truncated
            
            # Store transition and train
            agent.store_transition(state, action_indices, reward, next_state, done)
            agent.train_step()
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Update target network every 5 episodes
        if episode % 5 == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track rewards
        episode_rewards.append(episode_reward)
        
        # Calculate average
        window = min(10, len(episode_rewards))
        avg_reward = np.mean(episode_rewards[-window:])
        
        # Print progress
        print(f"       Reward: {episode_reward:7.2f} | "
              f"Avg(10): {avg_reward:7.2f} | "
              f"Steps: {episode_length:3d} | "
              f"ε: {agent.epsilon:.3f}")
        
        # Save best model
        if episode_reward > best_reward and episode > 10:
            best_reward = episode_reward
            agent.save("models/pose_trained_best.pt")
            print(f"       ✓ New best! Saved to models/pose_trained_best.pt")
        
        # Save checkpoint every 25 episodes
        if (episode + 1) % 25 == 0:
            agent.save(f"models/checkpoint_ep{episode+1}.pt")
            print(f"       ✓ Checkpoint saved\n")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total episodes: {episodes}")
    print(f"Poses used: {len(pose_library) if len(pose_library) > 0 else 'Default only'}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final avg reward: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"\nModel saved: models/pose_trained_best.pt")
    print("\nTo test the trained agent, run:")
    print("  python test_agent.py")
    
    env.close()


if __name__ == "__main__":
    main()