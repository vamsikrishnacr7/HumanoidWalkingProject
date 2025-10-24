"""
modules/module1_to_angles.py

Map BODY25 skeleton to lower-body joint angles.
BODY25 keypoint indices:
    0: Nose
    1: Neck
    2,5: Right/Left Shoulders
    3,6: Right/Left Elbows
    4,7: Right/Left Wrists
    8: MidHip
    9,12: Right/Left Hip
    10,13: Right/Left Knee
    11,14: Right/Left Ankle
    15,16: Right/Left Eye
    17,18: Right/Left Ear
    19-22: Left Big Toe, Small Toe, Heel
    22-25: Right Big Toe, Small Toe, Heel

Return order: [r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle]
"""

import numpy as np
from math import atan2, pi
import cv2

def safe_point(p):
    """Return 2D np.array or None if low confidence."""
    if p is None:
        return None
    x,y,c = p
    return np.array([x,y], dtype=float) if c > 0.1 else None

def angle_three(a,b,c):
    """Angle at b formed by a-b-c (signed)."""
    a = np.array(a, float); b = np.array(b, float); c = np.array(c, float)
    v1 = a - b; v2 = c - b
    if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
        return 0.0
    ang = atan2(v2[1], v2[0]) - atan2(v1[1], v1[0])
    # normalize
    while ang <= -pi: ang += 2*pi
    while ang > pi: ang -= 2*pi
    return float(ang)

def draw_skeleton_angles(image, body25, angles=None):
    """
    Draw skeleton with joint angles visualized.
    Args:
        image: RGB/BGR image to draw on
        body25: (25,3) skeleton array
        angles: Optional angles from compute_lower_body_angles
    Returns:
        Annotated image copy
    """
    vis = image.copy()
    if angles is None:
        angles = compute_lower_body_angles(body25)
    
    # Draw main skeleton lines
    pairs = [
        # Legs
        (8,9), (9,10), (10,11),  # right leg
        (8,12), (12,13), (13,14),  # left leg
        # Upper body
        (1,8),  # spine
        (1,2), (2,3), (3,4),  # right arm
        (1,5), (5,6), (6,7),  # left arm
    ]
    
    # Draw skeleton lines
    for i1,i2 in pairs:
        p1 = safe_point(body25[i1])
        p2 = safe_point(body25[i2])
        if p1 is not None and p2 is not None:
            p1 = tuple(map(int, p1))
            p2 = tuple(map(int, p2))
            cv2.line(vis, p1, p2, (0,255,0), 2)
    
    # Draw joint angles
    angles_dict = angles if isinstance(angles, dict) else None
    if angles_dict is None:
        angles_dict = compute_full_body_angles(body25)
    
    # Points for angle visualization (joint1, joint2, joint3, angle_name)
    angle_points = [
        # Lower body
        (9,10,11, 'r_knee'),    # right knee
        (12,13,14, 'l_knee'),   # left knee
        (8,9,10, 'r_hip'),      # right hip
        (8,12,13, 'l_hip'),     # left hip
        # Upper body
        (1,2,3, 'r_shoulder'),  # right shoulder
        (1,5,6, 'l_shoulder'),  # left shoulder
        (2,3,4, 'r_elbow'),     # right elbow
        (5,6,7, 'l_elbow'),     # left elbow
    ]
    
    for i1,i2,i3,ang_idx in angle_points:
        pts = [safe_point(body25[i]) for i in (i1,i2,i3)]
        if all(p is not None for p in pts):
            # Draw angle arc
            center = tuple(map(int, pts[1]))  # middle point
            angle = np.degrees(angles_dict[ang_idx])
            cv2.putText(vis, f"{angle:.1f}°", 
                       (center[0]+10, center[1]+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    
    return vis

def compute_full_body_angles(body25):
    """
    Compute full body joint angles from BODY25 skeleton.
    Args:
        body25: (25,3) array of (x,y,confidence) for each keypoint
    Returns:
        Dictionary with angles (in radians) for:
        - Lower body: r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle
        - Upper body: spine, neck, r_shoulder, r_elbow, l_shoulder, l_elbow
    """
    if body25 is None or body25.shape != (25,3):
        return np.zeros(6, dtype=np.float32)

    # indexes per our mapping earlier:
    # right hip=9, right knee=10, right ankle=11
    # left hip=12, left knee=13, left ankle=14
    def P(i): 
        return safe_point(body25[i]) if body25 is not None else None

    r_hip = r_knee = r_ankle = 0.0
    l_hip = l_knee = l_ankle = 0.0

    # Right knee angle: hip(9)-knee(10)-ankle(11)
    a,b,c = P(9), P(10), P(11)
    if a is not None and b is not None and c is not None:
        r_knee = angle_three(a,b,c)

    # Left knee
    a,b,c = P(12), P(13), P(14)
    if a is not None and b is not None and c is not None:
        l_knee = angle_three(a,b,c)

    # Right hip: neck/midhip - hip - knee (approx)
    midhip = safe_point(body25[8])  # midhip
    neck = safe_point(body25[1])
    a,b,c = midhip if midhip is not None else neck, P(9), P(10)
    if a is not None and b is not None and c is not None:
        r_hip = angle_three(a,b,c)

    # Left hip
    a,b,c = midhip if midhip is not None else neck, P(12), P(13)
    if a is not None and b is not None and c is not None:
        l_hip = angle_three(a,b,c)

    # Ankles: knee-ankle-toe approximation (use ankles+toes if available)
    # body25 toe indices: 19,20 (left), 22,23 (right) — approximate
    r_toe = safe_point(body25[22]) if body25.shape[0] > 22 else None
    l_toe = safe_point(body25[19]) if body25.shape[0] > 19 else None

    a,b,c = P(10), P(11), r_toe
    if a is not None and b is not None and c is not None:
        r_ankle = angle_three(a,b,c)
    a,b,c = P(13), P(14), l_toe
    if a is not None and b is not None and c is not None:
        l_ankle = angle_three(a,b,c)

    # Upper body angles
    spine = neck_angle = 0.0
    r_shoulder = r_elbow = 0.0
    l_shoulder = l_elbow = 0.0
    
    # Spine angle from vertical
    if midhip is not None and neck is not None:
        # Use vertical reference (0,1) for spine angle
        vert = np.array([0, -1], dtype=float)  # pointing up
        spine_vec = neck - midhip
        if np.linalg.norm(spine_vec) > 1e-6:
            spine = atan2(spine_vec[1], spine_vec[0]) - atan2(vert[1], vert[0])
            while spine <= -pi: spine += 2*pi
            while spine > pi: spine -= 2*pi
    
    # Right arm
    if all(P(i) is not None for i in [1,2,3,4]):  # neck, shoulder, elbow, wrist
        # Shoulder relative to spine
        r_shoulder = angle_three(P(1), P(2), P(3))
        # Elbow angle
        r_elbow = angle_three(P(2), P(3), P(4))
    
    # Left arm
    if all(P(i) is not None for i in [1,5,6,7]):  # neck, shoulder, elbow, wrist
        # Shoulder relative to spine
        l_shoulder = angle_three(P(1), P(5), P(6))
        # Elbow angle
        l_elbow = angle_three(P(5), P(6), P(7))

    # Return both lower and upper body angles
    angles = {
        # Lower body
        'r_hip': r_hip, 'r_knee': r_knee, 'r_ankle': r_ankle,
        'l_hip': l_hip, 'l_knee': l_knee, 'l_ankle': l_ankle,
        # Upper body
        'spine': spine, 'neck': neck_angle,
        'r_shoulder': r_shoulder, 'r_elbow': r_elbow,
        'l_shoulder': l_shoulder, 'l_elbow': l_elbow
    }
    
    return angles
