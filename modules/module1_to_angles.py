"""
modules/module1_to_angles.py

Map BODY25 skeleton to lower-body joint angles:
- right_hip, right_knee, right_ankle
- left_hip, left_knee, left_ankle

Return order: [r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle]
If any landmark missing, fallback to 0.0 for that joint.
"""

import numpy as np
from math import atan2, pi

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

def compute_lower_body_angles(body25):
    """
    body25: (25,3) array. Returns 6-angle vector:
    [r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle]
    """
    if body25 is None:
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

    out = np.array([r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle], dtype=np.float32)
    return out
