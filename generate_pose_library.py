"""
generate_pose_library.py

Batch process multiple images to create a diverse library of initial poses.
This library will be used for training to ensure generalization.
"""

import os
import argparse
import numpy as np
from pathlib import Path
import sys

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from module1_pose import run_on_image
from module1_to_angles import compute_full_body_angles
from main_character_selector import select_main_character
from pose_to_pybullet import AngleToPyBulletMapper, validate_pose


def process_single_image(image_path, mapper, verbose=False):
    """
    Process a single image and extract PyBullet-compatible pose.
    
    Args:
        image_path: Path to image file
        mapper: AngleToPyBulletMapper instance
        verbose: Print debug info
    
    Returns:
        pose_dict: Dictionary of joint angles, or None if failed
    """
    if verbose:
        print(f"\nProcessing: {image_path}")
    
    try:
        # Run pose detection
        vis_img, people_data = run_on_image(image_path)
        
        if len(people_data) == 0:
            if verbose:
                print("  ⚠️  No people detected")
            return None
        
        # Select main character
        main_idx, main_skeleton, selection_info = select_main_character(
            people_data, 
            method='weighted'
        )
        
        if verbose:
            print(f"  ✓ Detected {len(people_data)} person(s)")
            print(f"    Main: Person {selection_info['main_person_id']}")
        
        # Compute 2D angles
        angles_2d = compute_full_body_angles(main_skeleton)
        
        # Convert to PyBullet format
        pb_angles = mapper.convert_angles(angles_2d, debug=False)
        
        # Validate
        is_valid, warnings = validate_pose(pb_angles)
        
        if not is_valid:
            if verbose:
                print("  ⚠️  Pose validation failed:")
                for w in warnings:
                    print(f"      {w}")
            return None
        
        if verbose:
            print(f"  ✓ Valid pose extracted")
        
        return pb_angles
        
    except Exception as e:
        if verbose:
            print(f"  ❌ Error: {e}")
        return None


def generate_pose_library(
    image_folder,
    output_file,
    view_type='front',
    max_poses=None,
    verbose=True
):
    """
    Generate pose library from all images in a folder.
    
    Args:
        image_folder: Path to folder containing images
        output_file: Output .npy file path
        view_type: Camera view type ('front', 'side', 'angled')
        max_poses: Maximum number of poses to generate (None = all)
        verbose: Print progress
    
    Returns:
        Number of poses successfully generated
    """
    print("="*70)
    print("POSE LIBRARY GENERATION")
    print("="*70)
    print(f"Image folder: {image_folder}")
    print(f"Output file: {output_file}")
    print(f"View type: {view_type}")
    print()
    
    # Create mapper
    mapper = AngleToPyBulletMapper(view_type=view_type)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_folder = Path(image_folder)
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_folder.glob(f'*{ext}')))
        image_files.extend(list(image_folder.glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images")
    
    if max_poses is not None:
        image_files = image_files[:max_poses]
        print(f"Processing first {max_poses} images")
    
    print()
    
    # Process all images
    pose_library = []
    successful = 0
    failed = 0
    
    for i, img_path in enumerate(image_files):
        if verbose:
            print(f"[{i+1}/{len(image_files)}] ", end="")
        
        pose = process_single_image(str(img_path), mapper, verbose=verbose)
        
        if pose is not None:
            pose_library.append(pose)
            successful += 1
        else:
            failed += 1
    
    # Save library
    print()
    print("="*70)
    print(f"RESULTS:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  Total poses: {len(pose_library)}")
    
    if len(pose_library) == 0:
        print("\n❌ No valid poses generated! Check your images.")
        return 0
    
    # Convert to structured array for saving
    # Save as list of dictionaries
    np.save(output_file, pose_library, allow_pickle=True)
    
    print(f"\n✓ Pose library saved to: {output_file}")
    print("="*70)
    
    return len(pose_library)


def load_pose_library(filepath):
    """
    Load pose library from .npy file.
    
    Args:
        filepath: Path to .npy file
    
    Returns:
        List of pose dictionaries
    """
    poses = np.load(filepath, allow_pickle=True)
    return list(poses)


def sample_random_pose(pose_library):
    """
    Sample a random pose from the library.
    
    Args:
        pose_library: List of pose dictionaries
    
    Returns:
        Random pose dictionary
    """
    return pose_library[np.random.randint(len(pose_library))]


def main():
    parser = argparse.ArgumentParser(description="Generate pose library for RL training")
    
    parser.add_argument(
        '--image-folder',
        type=str,
        required=True,
        help='Folder containing pose images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='pose_library.npy',
        help='Output .npy file path (default: pose_library.npy)'
    )
    
    parser.add_argument(
        '--view',
        type=str,
        default='front',
        choices=['front', 'side', 'angled'],
        help='Camera view type (default: front)'
    )
    
    parser.add_argument(
        '--max-poses',
        type=int,
        default=None,
        help='Maximum number of poses to process (default: all)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Generate library
    num_poses = generate_pose_library(
        image_folder=args.image_folder,
        output_file=args.output,
        view_type=args.view,
        max_poses=args.max_poses,
        verbose=not args.quiet
    )
    
    if num_poses > 0:
        print(f"\n✓ Success! Generated {num_poses} poses")
        print(f"\nYou can now use this library for training:")
        print(f"  python train.py --pose-library {args.output}")
    else:
        print("\n❌ Failed to generate pose library")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())