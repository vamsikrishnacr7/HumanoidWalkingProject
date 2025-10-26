"""
test.py

Complete end-to-end pipeline:
1. Detect skeleton (YOLO + MediaPipe)
2. Compute joint angles (2D image space)
3. Convert to PyBullet angles (3D joint space)
4. Load and stabilize humanoid in PyBullet
"""

import sys
import time
import numpy as np
from pathlib import Path

# Your existing modules
from modules.module1_pose import run_on_image, get_random_image
from modules.module1_to_angles import compute_full_body_angles
from modules.main_character_selector import select_main_character

# New mapping module
from modules.pose_to_pybullet import (
    AngleToPyBulletMapper, 
    enhanced_reset_with_pose,
    safe_initial_pose,
    validate_pose
)

# Your environment
from modules.env_humanoid import HumanoidWalkEnv


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pose → PyBullet Pipeline")
    parser.add_argument("--input", type=str, help="Path to input image")
    parser.add_argument("--folder", type=str, default="poses/sample_images",
                       help="Folder for random image selection")
    parser.add_argument("--view", type=str, default="front", 
                       choices=['front', 'side', 'angled'],
                       help="Camera view type (affects hip angle interpretation)")
    parser.add_argument("--safe-mode", action="store_true",
                       help="Use safe default pose instead of detected pose")
    parser.add_argument("--sim-time", type=int, default=5,
                       help="Simulation time in seconds (default: 5)")
    args = parser.parse_args()
    
    # =======================================================================
    # STEP 1: Get image and detect pose
    # =======================================================================
    print("="*70)
    print("STEP 1: POSE DETECTION")
    print("="*70)
    
    if args.input:
        image_path = args.input
    else:
        print(f"Selecting random image from '{args.folder}'...")
        image_path = get_random_image(args.folder)
    
    print(f"Processing: {image_path}\n")
    
    # Run pose detection
    vis_img, people_data = run_on_image(image_path)
    
    if len(people_data) == 0:
        print("❌ No people detected in image!")
        if not args.safe_mode:
            print("   Try using --safe-mode to load default pose")
            return
        else:
            print("   Loading safe default pose instead...")
            detected_angles = None
    else:
        # Select main character
        main_idx, main_skeleton, selection_info = select_main_character(
            people_data, 
            method='weighted'
        )
        
        print(f"\n✓ Detected {len(people_data)} person(s)")
        print(f"  Main character: Person {selection_info['main_person_id']}")
        print(f"  Bbox area: {selection_info['bbox_area']}")
        print(f"  Confidence: {selection_info['confidence']:.2f}\n")
        
        # Compute angles from skeleton
        detected_angles = compute_full_body_angles(main_skeleton)
        
        print("Detected joint angles:")
        for joint in ['r_hip', 'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle']:
            angle_deg = np.degrees(detected_angles[joint])
            print(f"  {joint.upper():<10}: {angle_deg:>7.1f}°")
    
    # =======================================================================
    # STEP 2: Convert angles to PyBullet format
    # =======================================================================
    print("\n" + "="*70)
    print("STEP 2: ANGLE CONVERSION")
    print("="*70)
    
    mapper = AngleToPyBulletMapper(view_type=args.view)
    
    if args.safe_mode or detected_angles is None:
        print("Using safe default pose...")
        pb_angles = safe_initial_pose()
        print("\nSafe pose angles:")
        for joint, angle in pb_angles.items():
            print(f"  {joint:<15}: {np.degrees(angle):>7.1f}°")
    else:
        print(f"Converting angles (view: {args.view})...\n")
        pb_angles = mapper.convert_angles(detected_angles, debug=True)
        
        # Validate converted pose
        is_valid, warnings = validate_pose(pb_angles)
        if warnings:
            print("⚠️  VALIDATION WARNINGS:")
            for w in warnings:
                print(f"   {w}")
            print()
    
    # =======================================================================
    # STEP 3: Load in PyBullet
    # =======================================================================
    print("="*70)
    print("STEP 3: PYBULLET SIMULATION")
    print("="*70)
    
    try:
        # Initialize environment
        print("Initializing PyBullet environment...")
        env = HumanoidWalkEnv(urdf_path="data/humanoid.urdf")
        
        # Use enhanced reset with pose
        print("\nLoading humanoid with pose...")
        obs = enhanced_reset_with_pose(env, pb_angles, stabilization_steps=200)
        
        print(f"\n✓ Humanoid loaded successfully")
        print(f"  Observation shape: {obs.shape}")
        
        # =======================================================================
        # STEP 4: Run simulation
        # =======================================================================
        print("\n" + "="*70)
        print(f"STEP 4: RUNNING SIMULATION ({args.sim_time}s)")
        print("="*70)
        print("Controls:")
        print("  - Watch the humanoid maintain the pose")
        print("  - Close window or press Ctrl+C to exit\n")
        
        sim_steps = args.sim_time * 240  # 240Hz
        
        for step in range(sim_steps):
            # Step simulation (no action for now - just holding pose)
            obs, reward, terminated, truncated, info = env.step()
            
            # Print status every second
            if step % 240 == 0:
                import pybullet as p
                pos, orn = p.getBasePositionAndOrientation(env.robot)
                rpy = p.getEulerFromQuaternion(orn)
                print(f"  t={step/240:.1f}s: Z={pos[2]:.3f}m, "
                      f"Roll={np.degrees(rpy[0]):>5.1f}°, "
                      f"Pitch={np.degrees(rpy[1]):>5.1f}°")
            
            time.sleep(1./240)
            
            # Check for termination
            if terminated:
                print("\n⚠️  Simulation terminated (robot fell)")
                break
        
        print("\n✓ Simulation complete")
        
        # Keep window open
        print("\nSimulation paused. Close window to exit...")
        try:
            while True:
                env.step()
                time.sleep(1./240)
        except KeyboardInterrupt:
            print("\nExiting...")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("   Make sure your URDF file exists at 'data/humanoid.urdf'")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            import pybullet as p
            p.disconnect()
            print("PyBullet disconnected")
        except:
            pass


if __name__ == "__main__":
    main()