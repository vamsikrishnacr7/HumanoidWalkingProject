"""
modules/pose_to_pybullet.py

SIMPLIFIED VERSION - For basic 1-DOF per joint URDF
Key fix: For standing, ALL joints should be at 0° (straight legs)
"""

import numpy as np
from math import pi, degrees, radians


class AngleToPyBulletMapper:
    """Maps 2D pose angles to PyBullet joint conventions."""
    
    def __init__(self, view_type='front'):
        self.view_type = view_type
        
        self.joint_limits = {
            'right_hip': (-0.5, 1.5),
            'right_knee': (0.0, 2.0),
            'right_ankle': (-0.4, 0.4),
            'left_hip': (-0.5, 1.5),
            'left_knee': (0.0, 2.0),
            'left_ankle': (-0.4, 0.4),
        }
    
    def convert_angles(self, angles_dict, debug=False):
        """
        Convert 2D pose angles to PyBullet format.
        
        KEY PRINCIPLE: For STANDING pose, everything should be 0° (straight)
        Only add angles for actual movements (sitting, squatting, etc.)
        """
        pb_angles = {}
        
        # Get knee angles
        r_knee_2d = angles_dict.get('r_knee', 0.0)
        l_knee_2d = angles_dict.get('l_knee', 0.0)
        
        # === KNEES ===
        pb_angles['right_knee'] = self._convert_knee(r_knee_2d)
        pb_angles['left_knee'] = self._convert_knee(l_knee_2d)
        
        # === HIPS ===
        # CRITICAL: For standing, hips should be 0° (straight)
        # Only bend if knees are significantly bent (squatting)
        pb_angles['right_hip'] = self._convert_hip(pb_angles['right_knee'])
        pb_angles['left_hip'] = self._convert_hip(pb_angles['left_knee'])
        
        # === ANKLES ===
        # CRITICAL: For standing, ankles MUST be 0° (flat feet)
        pb_angles['right_ankle'] = 0.0
        pb_angles['left_ankle'] = 0.0
        
        if debug:
            self._print_conversion(angles_dict, pb_angles)
        
        return pb_angles
    
    def _convert_knee(self, angle_2d):
        """
        Convert knee angle.
        
        For STANDING: knee should be 0° (straight)
        Only bend if clearly bent in image
        """
        # If angle is small, keep straight
        if abs(angle_2d) < 0.5:  # Less than ~28°
            return 0.0  # PERFECTLY STRAIGHT
        
        # If clearly bent (angle < -0.5)
        if angle_2d < -0.5:
            flexion = abs(angle_2d)
            # Reduce to conservative value
            flexion = min(flexion * 0.6, 1.5)  # Max ~86°
            return flexion
        
        return 0.0  # Default: straight
    
    def _convert_hip(self, knee_angle):
        """
        Convert hip angle based on knee bend.
        
        CRITICAL: For standing (knee ~0°), hip MUST be 0°
        Only flex hip when squatting deeply
        """
        # If standing (straight knees)
        if knee_angle < 0.3:  # Less than ~17° knee bend
            return 0.0  # PERFECTLY STRAIGHT HIP
        
        # If squatting (bent knees)
        if knee_angle > 1.0:  # More than ~57° knee bend
            # Hip needs to flex too
            hip_flex = knee_angle * 0.3  # Proportional to knee
            return min(hip_flex, 0.8)  # Max ~46°
        
        # Slight squat
        return knee_angle * 0.15
    
    def _print_conversion(self, angles_2d, angles_pb):
        """Print conversion details."""
        print("\n" + "="*70)
        print("ANGLE CONVERSION: 2D Pose → PyBullet")
        print("="*70)
        print(f"{'Joint':<15} {'2D Input':<15} {'→':<3} {'PyBullet':<15}")
        print("-"*70)
        
        mapping = {
            'right_hip': 'r_hip',
            'right_knee': 'r_knee',
            'right_ankle': 'r_ankle',
            'left_hip': 'l_hip',
            'left_knee': 'l_knee',
            'left_ankle': 'l_ankle',
        }
        
        for pb_name, pose_name in mapping.items():
            angle_2d = angles_2d.get(pose_name, 0.0)
            angle_pb = angles_pb.get(pb_name, 0.0)
            
            print(f"{pb_name:<15} {degrees(angle_2d):>7.1f}°{' '*6} "
                  f"→   {degrees(angle_pb):>7.1f}°")
        
        print("="*70)
        print("NOTE: For standing, all angles forced to 0° (straight)")
        print()


def safe_initial_pose():
    """
    Safe standing pose - ALL JOINTS AT 0° (STRAIGHT).
    This is the MOST STABLE configuration.
    """
    return {
        'right_hip': 0.0,      # Straight
        'right_knee': 0.0,     # Straight
        'right_ankle': 0.0,    # Flat
        'left_hip': 0.0,       # Straight
        'left_knee': 0.0,      # Straight
        'left_ankle': 0.0,     # Flat
    }


def validate_pose(angles_dict):
    """Check if pose is physically plausible."""
    warnings = []
    
    for joint, angle in angles_dict.items():
        if abs(angle) > pi:
            warnings.append(f"{joint}: {degrees(angle):.1f}° exceeds ±180°")
    
    # Check for asymmetry
    if 'right_knee' in angles_dict and 'left_knee' in angles_dict:
        diff = abs(angles_dict['right_knee'] - angles_dict['left_knee'])
        if diff > radians(30):
            warnings.append(f"Knee asymmetry: {degrees(diff):.1f}°")
    
    return len(warnings) == 0, warnings


def enhanced_reset_with_pose(env_instance, angles_dict, stabilization_steps=500, use_motors=True):
    """
    Enhanced reset with MAXIMUM stability focus.
    """
    import pybullet as p
    
    # Validate
    is_valid, warnings = validate_pose(angles_dict)
    if warnings:
        print("⚠️  WARNINGS:")
        for w in warnings:
            print(f"   {w}")
    
    # Reset
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1./240.)
    
    # Ground with HIGH friction
    env_instance.plane = p.loadURDF("plane.urdf")
    p.changeDynamics(env_instance.plane, -1, 
                     lateralFriction=2.0,
                     spinningFriction=0.5,
                     rollingFriction=0.1)
    
    # Robot height: pelvis(0.15) + thigh(0.40) + shin(0.40) + foot(0.03) = 0.98m
    start_height = 1.00  # Slightly higher to avoid ground penetration
    
    # CRITICAL: Perfect upright orientation
    upright = p.getQuaternionFromEuler([0, 0, 0])
    
    env_instance.robot = p.loadURDF(
        env_instance.urdf_path,
        [0, 0, start_height],
        upright,
        useFixedBase=False,
        flags=p.URDF_USE_SELF_COLLISION
    )
    
    print(f"[INFO] Robot loaded at Z={start_height}m")
    
    # Joint map
    joint_map = {}
    for j in range(p.getNumJoints(env_instance.robot)):
        info = p.getJointInfo(env_instance.robot, j)
        jname = info[1].decode('utf-8')
        joint_map[jname] = j
    
    # CRITICAL: Add MAXIMUM friction to feet
    for j in range(p.getNumJoints(env_instance.robot)):
        info = p.getJointInfo(env_instance.robot, j)
        link_name = info[12].decode('utf-8')
        
        if 'foot' in link_name.lower():
            p.changeDynamics(env_instance.robot, j,
                           lateralFriction=3.0,      # VERY HIGH
                           spinningFriction=1.0,
                           rollingFriction=0.3,
                           restitution=0.0,
                           contactStiffness=30000,   # Stiff contact
                           contactDamping=1000)
            print(f"  ✓ Max friction on {link_name}")
    
    # Apply joint angles
    print("[INFO] Setting joint angles:")
    for jname, angle in angles_dict.items():
        if jname in joint_map:
            j_idx = joint_map[jname]
            p.resetJointState(env_instance.robot, j_idx, angle)
            print(f"  ✓ {jname:<15} = {degrees(angle):>6.1f}°")
    
    # CRITICAL: VERY strong motors
    if use_motors:
        for jname, angle in angles_dict.items():
            if jname in joint_map:
                p.setJointMotorControl2(
                    env_instance.robot,
                    joint_map[jname],
                    p.POSITION_CONTROL,
                    targetPosition=angle,
                    force=1000,           # VERY HIGH force
                    positionGain=0.8,     # Strong position control
                    velocityGain=0.5,
                    maxVelocity=1.0       # Slow movements
                )
    
    # CRITICAL: High damping on base to prevent tipping
    p.changeDynamics(env_instance.robot, -1, 
                     linearDamping=0.5,    # High damping
                     angularDamping=1.0)
    
    # Long stabilization
    print(f"[INFO] Stabilizing for {stabilization_steps} steps...")
    
    for i in range(stabilization_steps):
        # Refresh motors every 20 steps
        if i % 20 == 0 and use_motors:
            for jname, angle in angles_dict.items():
                if jname in joint_map:
                    p.setJointMotorControl2(
                        env_instance.robot,
                        joint_map[jname],
                        p.POSITION_CONTROL,
                        targetPosition=angle,
                        force=1000,
                        positionGain=0.8,
                        velocityGain=0.5
                    )
        
        p.stepSimulation()
        
        # Monitor
        if i % 100 == 0:
            pos, orn = p.getBasePositionAndOrientation(env_instance.robot)
            rpy = p.getEulerFromQuaternion(orn)
            
            print(f"  Step {i:3d}: Z={pos[2]:.3f}m, "
                  f"Roll={degrees(rpy[0]):>5.1f}°, "
                  f"Pitch={degrees(rpy[1]):>5.1f}°")
            
            if pos[2] < 0.3:
                print("  ⚠️  Robot falling!")
                break
    
    # Final check
    final_pos, final_orn = p.getBasePositionAndOrientation(env_instance.robot)
    final_rpy = p.getEulerFromQuaternion(final_orn)
    
    print(f"\n[RESULT]")
    print(f"  Z-height: {final_pos[2]:.3f}m")
    print(f"  Tilt: Roll={degrees(final_rpy[0]):.1f}°, Pitch={degrees(final_rpy[1]):.1f}°")
    
    if final_pos[2] > 0.7:
        print("  ✓✓✓ STABLE!")
    else:
        print("  ❌ COLLAPSED")
        print("\n  Try this:")
        print("  1. Check URDF foot origins are at 'xyz=\"0.03 0 -0.015\"'")
        print("  2. Check both ankle joints are at z=-0.415")
        print("  3. Make sure you're using the updated URDF from earlier")
    
    return env_instance.get_observation()


if __name__ == "__main__":
    print("SIMPLE STANDING POSE MAPPER")
    print("="*70)
    print("Strategy: Keep ALL joints at 0° for perfect standing stability")
    print("="*70)