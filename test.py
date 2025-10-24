"""
test.py

End-to-end:
1. Detect skeleton (YOLO+MediaPipe)
2. Compute joint angles
3. Display humanoid with angles
"""

from modules.module1_pose import run_on_image
from modules.module1_to_angles import compute_joint_angles
from envs.humanoid_env import HumanoidWalkEnv
import time

def main():
    print("Running Pose Detection...")
    skels = run_on_image("poses/sample_images/test_pose.png")

    if not skels:
        print("No skeleton found!")
        return

    skel = skels[0]
    print("Computing Angles...")
    angles = compute_joint_angles(skel)
    print("Angles:", angles)

    print("Simulating Humanoid...")
    env = HumanoidWalkEnv(gui=True)
    env.reset(initial_pose=angles)

    for _ in range(600):
        env.step()
        time.sleep(1./240)

    env.close()

if __name__ == "__main__":
    main()
