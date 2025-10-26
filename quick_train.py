"""
quick_train.py - Quick training to see agent move (IMPROVED)

REPLACE your existing quick_train.py with this version.

Usage:
    python quick_train.py
"""

import numpy as np
from pathlib import Path

from modules.env_humanoid import HumanoidWalkEnv
from modules.dqn_agent import DQNAgent


def main():
    print("="*70)
    print("QUICK TRAINING - Humanoid Learning to Walk")
    print("="*70)
    print("\nThis will train for 100 episodes.")
    print("Watch the PyBullet window - the humanoid will learn!\n")
    print("What to expect:")
    print("  Episodes 1-20:  Falls quickly (exploring)")
    print("  Episodes 20-50: Stays up longer")
    print("  Episodes 50+:   Should move forward!\n")
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    
    # Initialize environment (with rendering)
    env = HumanoidWalkEnv(urdf_path="data/humanoid.urdf", render=True)
    
    # Get state dimension
    dummy_obs = env.reset()
    state_dim = dummy_obs.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Control joints: {len(env.CONTROL_JOINTS)}\n")
    
    # Initialize agent with BETTER hyperparameters
    agent = DQNAgent(
        state_dim=state_dim, 
        num_joints=4, 
        actions_per_joint=5,
        learning_rate=3e-4,      # Higher learning rate
        epsilon_start=0.9,        # Start with less randomness
        epsilon_decay=0.99        # Slower decay
    )
    
    # Training
    episodes = 100  # More episodes!
    print(f"Starting training for {episodes} episodes...\n")
    
    episode_rewards = []
    best_reward = -float('inf')  # FIXED: was -999
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Run episode
        while not done and episode_length < 1000:
            # Select action
            action_indices, torques = agent.select_action(state, training=True)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(torques)
            done = terminated or truncated
            
            # Store transition and train
            agent.store_transition(state, action_indices, reward, next_state, done)
            loss = agent.train_step()
            
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
        status = ""
        if episode_reward > best_reward:
            status = " 🌟 NEW BEST!"
            best_reward = episode_reward
            agent.save("models/quick_train_best.pt")
        
        print(f"Ep {episode + 1:3d}/{episodes} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Avg(10): {avg_reward:7.2f} | "
              f"Steps: {episode_length:3d} | "
              f"ε: {agent.epsilon:.3f}{status}")
        
        # Print progress milestones
        if episode == 19:
            print(f"\n{'─'*70}")
            print("20 episodes done! Agent should start staying up longer now...")
            print(f"{'─'*70}\n")
        elif episode == 49:
            print(f"\n{'─'*70}")
            print("50 episodes done! Agent should start moving forward soon...")
            print(f"{'─'*70}\n")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Total episodes: {episodes}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final avg reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"\nModel saved: models/quick_train_best.pt")
    
    # Advice based on results
    if best_reward > 30:
        print("\n✅ GREAT! Your agent learned to walk!")
    elif best_reward > 10:
        print("\n⚠️  OKAY - Agent is learning but needs more training.")
        print("   Try running for 200 episodes or adjust reward weights.")
    else:
        print("\n❌ Agent struggled to learn.")
        print("   This could mean:")
        print("   1. Need more episodes (try 200)")
        print("   2. URDF might be unstable")
        print("   3. Reward weights need tuning")
    
    print("\nTo test the trained agent, run:")
    print("  python test_agent.py\n")
    
    env.close()


if __name__ == "__main__":
    main()