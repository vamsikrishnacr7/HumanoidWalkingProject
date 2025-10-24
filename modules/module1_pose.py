import cv2
import numpy as np
import mediapipe as mp

# === BODY25 mapping (your original) ===
BODY25 = {
    "Nose":0, "Neck":1, "RShoulder":2, "RElbow":3, "RWrist":4,
    "LShoulder":5, "LElbow":6, "LWrist":7, "MidHip":8,
    "RHip":9, "RKnee":10, "RAnkle":11, "LHip":12, "LKnee":13,
    "LAnkle":14, "REye":15, "LEye":16, "REar":17, "LEar":18,
    "LBigToe":19, "LSmallToe":20, "LHeel":21, "RBigToe":22,
    "RSmallToe":23, "RHeel":24
}

BODY25_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(10,11),(8,12),(12,13),(13,14),
    (0,15),(15,17),(0,16),(16,18),(14,19),(19,20),(14,21),
    (11,22),(22,23),(11,24)
]

mp_pose = mp.solutions.pose

def extract_skeleton(image):
    """Extract full 33 landmark skeleton from MediaPipe Pose."""
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,  # More accurate
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        h, w, _ = image.shape
        skeleton = np.zeros((33, 3), dtype=np.float32)
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            skeleton[idx, 0] = lm.x * w
            skeleton[idx, 1] = lm.y * h
            skeleton[idx, 2] = lm.visibility
        return skeleton

def remap_to_body25(skel33):
    """Optional: remap MediaPipe 33-point skeleton to BODY25 format."""
    # Basic mapping (some BODY25 points like 'Neck' are approximated)
    body25 = np.zeros((25, 3), dtype=np.float32)
    # Nose
    body25[0] = skel33[0]
    # Neck approx: midpoint between shoulders
    body25[1,:2] = (skel33[11,:2] + skel33[12,:2]) / 2
    body25[1,2] = (skel33[11,2] + skel33[12,2]) / 2
    # R shoulder, elbow, wrist
    body25[2] = skel33[12]
    body25[3] = skel33[14]
    body25[4] = skel33[16]
    # L shoulder, elbow, wrist
    body25[5] = skel33[11]
    body25[6] = skel33[13]
    body25[7] = skel33[15]
    # MidHip approx: midpoint between hips
    body25[8,:2] = (skel33[23,:2] + skel33[24,:2]) / 2
    body25[8,2] = (skel33[23,2] + skel33[24,2]) / 2
    # R hip, knee, ankle
    body25[9] = skel33[24]
    body25[10] = skel33[26]
    body25[11] = skel33[28]
    # L hip, knee, ankle
    body25[12] = skel33[23]
    body25[13] = skel33[25]
    body25[14] = skel33[27]
    # Eyes, ears
    body25[15] = skel33[2]
    body25[16] = skel33[5]
    body25[17] = skel33[4]
    body25[18] = skel33[7]
    # Toes and heels
    body25[19] = skel33[31]
    body25[20] = skel33[32]
    body25[21] = skel33[30]
    body25[22] = skel33[28]
    body25[23] = skel33[29]
    body25[24] = skel33[27]
    return body25

def visualize_skeleton(image, skeleton):
    img = image.copy()
    for i, j in BODY25_CONNECTIONS:
        if skeleton[i, 2] > 0.1 and skeleton[j, 2] > 0.1:
            pt1 = tuple(skeleton[i, :2].astype(int))
            pt2 = tuple(skeleton[j, :2].astype(int))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
    for x, y, c in skeleton:
        if c > 0.1:
            cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
    return img

if __name__ == "__main__":
    img_path = "poses/sample_images/test_pose.png"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError("Image not found!")

    # Upscale image for better detection
    img = cv2.resize(img, (512, int(img.shape[0] * 512 / img.shape[1])))

    skel33 = extract_skeleton(img)
    if skel33 is None:
        print("No pose detected.")
    else:
        # Debug: check visibility of lower body
        print("Landmark visibilities:")
        for i, v in enumerate(skel33[:, 2]):
            print(f"{i:2d}: {v:.3f}")

        # Convert to BODY25
        skel25 = remap_to_body25(skel33)

        vis_img = visualize_skeleton(img, skel25)
        cv2.imshow("Improved Skeleton Detection", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
"""
modules/module1_pose.py

Multi-person pose extraction:
1. YOLO person detection
2. MediaPipe Pose on each person crop
3. Fallback: if YOLO fails → full image skeleton
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path

BODY25_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(10,11),(8,12),(12,13),(13,14),
    (0,15),(15,17),(0,16),(16,18),(14,19),(19,20),(14,21),
    (11,22),(22,23),(11,24)
]

mp_pose = mp.solutions.pose
YOLO_MODEL = "yolov8n.pt"


def detect_person_bboxes(image, conf_thres=0.15):
    model = YOLO(YOLO_MODEL)
    results = model(image, conf=conf_thres, iou=0.7, classes=[0], verbose=False)
    boxes = []
    for r in results:
        if not r.boxes:
            continue
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            conf = float(b.conf[0])
            x1,y1,x2,y2 = xyxy
            boxes.append({'bbox':(x1,y1,x2,y2), 'conf':conf})
    return boxes


def extract_skel_33(image):
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as pose:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if not res.pose_landmarks:
            return None

        h,w,_ = image.shape
        sk = np.zeros((33,3), np.float32)
        for i,lm in enumerate(res.pose_landmarks.landmark):
            sk[i,0] = lm.x*w
            sk[i,1] = lm.y*h
            sk[i,2] = lm.visibility
        return sk


def remap_to_body25(sk):
    body25 = np.zeros((25,3), np.float32)
    body25[0] = sk[0]
    body25[1,:2] = (sk[11,:2]+sk[12,:2])/2
    body25[1,2] = (sk[11,2]+sk[12,2])/2

    body25[2] = sk[12]; body25[3] = sk[14]; body25[4] = sk[16]
    body25[5] = sk[11]; body25[6] = sk[13]; body25[7] = sk[15]

    body25[8,:2] = (sk[23,:2]+sk[24,:2])/2
    body25[8,2] = (sk[23,2]+sk[24,2])/2

    body25[9] = sk[24]; body25[10] = sk[26]; body25[11] = sk[28]
    body25[12] = sk[23]; body25[13] = sk[25]; body25[14] = sk[27]

    body25[15] = sk[2];  body25[16] = sk[5]
    body25[17] = sk[4];  body25[18] = sk[7]

    body25[19] = sk[31]; body25[20] = sk[32]
    body25[21] = sk[30]
    body25[22] = sk[28]; body25[23] = sk[29]
    body25[24] = sk[27]
    return body25


def draw_skel(img, sk):
    for i,j in BODY25_CONNECTIONS:
        if sk[i,2]>0.1 and sk[j,2]>0.1:
            p1 = tuple(sk[i,:2].astype(int))
            p2 = tuple(sk[j,:2].astype(int))
            cv2.line(img, p1, p2, (0,255,0), 2)

    for x,y,c in sk:
        if c>0.1:
            cv2.circle(img,(int(x),int(y)),4,(0,0,255),-1)


def run_on_image(path, out="outputs/pose_vis.png"):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError("Image not found!")

    # Upscale for accuracy
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))

    boxes = detect_person_bboxes(img)
    print(f"YOLO detected {len(boxes)} person(s)")

    results = []
    vis = img.copy()

    if len(boxes) == 0:
        print("Fallback: full-frame MediaPipe")
        sk33 = extract_skel_33(img)
        if sk33 is None:
            print("No pose found at all.")
            return []
        sk25 = remap_to_body25(sk33)
        draw_skel(vis, sk25)
        cv2.imwrite(out, vis)
        print(f"Saved {out}")
        return [sk25]

    for i,b in enumerate(boxes):
        x1,y1,x2,y2 = b['bbox']
        pad = 40
        x1 = max(0, x1-pad); y1 = max(0, y1-pad)
        x2 = min(img.shape[1], x2+pad); y2 = min(img.shape[0], y2+pad)

        crop = img[y1:y2, x1:x2]
        sk33 = extract_skel_33(crop)
        if sk33 is None:
            print(f"Person {i}: No pose")
            continue

        sk25_crop = remap_to_body25(sk33)

        sk25 = sk25_crop.copy()
        sk25[:,0]+=x1; sk25[:,1]+=y1

        draw_skel(vis, sk25)
        results.append(sk25)

        print(f"Person {i}: pose OK, conf {b['conf']:.2f}")

    Path(out).parent.mkdir(exist_ok=True)
    cv2.imwrite(out, vis)
    print(f"Saved {out}")

    return results


if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="outputs/pose_vis.png")
    args = ap.parse_args()
    run_on_image(args.input, args.out)
