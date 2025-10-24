"""
modules/module1_pose.py

Multi-person pose extraction:
1. YOLO person detection
2. MediaPipe Pose on each person crop
3. Fallback: if YOLO fails → full image skeleton
4. Display results using matplotlib
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import random

BODY25_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(10,11),(8,12),(12,13),(13,14),
    (0,15),(15,17),(0,16),(16,18),(14,19),(19,20),(14,21),
    (11,22),(22,23),(11,24)
]

mp_pose = mp.solutions.pose
YOLO_MODEL = "yolov8n.pt"


def detect_person_bboxes(image, conf_thres=0.15):
    """Detect person bounding boxes using YOLO."""
    model = YOLO(YOLO_MODEL)
    results = model(image, conf=conf_thres, iou=0.7, classes=[0], verbose=False)
    boxes = []
    for r in results:
        if not r.boxes:
            continue
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            conf = float(b.conf[0])
            x1, y1, x2, y2 = xyxy
            boxes.append({'bbox': (x1, y1, x2, y2), 'conf': conf})
    return boxes


def extract_skel_33(image):
    """Extract 33-point MediaPipe skeleton."""
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

        h, w, _ = image.shape
        sk = np.zeros((33, 3), np.float32)
        for i, lm in enumerate(res.pose_landmarks.landmark):
            sk[i, 0] = lm.x * w
            sk[i, 1] = lm.y * h
            sk[i, 2] = lm.visibility
        return sk


def remap_to_body25(sk):
    """Remap 33-point skeleton to BODY25 format."""
    body25 = np.zeros((25, 3), np.float32)
    body25[0] = sk[0]
    body25[1, :2] = (sk[11, :2] + sk[12, :2]) / 2
    body25[1, 2] = (sk[11, 2] + sk[12, 2]) / 2

    body25[2] = sk[12]; body25[3] = sk[14]; body25[4] = sk[16]
    body25[5] = sk[11]; body25[6] = sk[13]; body25[7] = sk[15]

    body25[8, :2] = (sk[23, :2] + sk[24, :2]) / 2
    body25[8, 2] = (sk[23, 2] + sk[24, 2]) / 2

    body25[9] = sk[24]; body25[10] = sk[26]; body25[11] = sk[28]
    body25[12] = sk[23]; body25[13] = sk[25]; body25[14] = sk[27]

    body25[15] = sk[2]; body25[16] = sk[5]
    body25[17] = sk[4]; body25[18] = sk[7]

    body25[19] = sk[31]; body25[20] = sk[32]
    body25[21] = sk[30]
    body25[22] = sk[28]; body25[23] = sk[29]
    body25[24] = sk[27]
    return body25


def draw_skel(img, sk):
    """Draw skeleton on image (in-place)."""
    for i, j in BODY25_CONNECTIONS:
        if sk[i, 2] > 0.1 and sk[j, 2] > 0.1:
            p1 = tuple(sk[i, :2].astype(int))
            p2 = tuple(sk[j, :2].astype(int))
            cv2.line(img, p1, p2, (0, 255, 0), 2)

    for x, y, c in sk:
        if c > 0.1:
            cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)


def get_random_image(folder="../poses/sample_images"):
    """Get a random image from the specified folder."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder '{folder}' not found!")
    
    # Support common image formats
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']
    images = []
    for ext in image_extensions:
        images.extend(folder_path.glob(ext))
    
    if not images:
        raise FileNotFoundError(f"No images found in '{folder}'")
    
    selected = random.choice(images)
    print(f"Selected image: {selected}")
    return str(selected)


def run_on_image(path):
    """Process image and collect all detected people."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    # Upscale for accuracy
    img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))

    boxes = detect_person_bboxes(img)
    print(f"YOLO detected {len(boxes)} person(s)")

    people_data = []  # Store all people info
    vis = img.copy()

    if len(boxes) == 0:
        print("Fallback: full-frame MediaPipe")
        sk33 = extract_skel_33(img)
        if sk33 is None:
            print("No pose found at all.")
            return img, []
        sk25 = remap_to_body25(sk33)
        draw_skel(vis, sk25)
        
        # Full image becomes the only detection
        people_data.append({
            'skeleton': sk25,
            'bbox_info': {
                'bbox': (0, 0, img.shape[1], img.shape[0]), 
                'conf': 1.0, 
                'area': img.shape[0] * img.shape[1]
            },
            'person_id': 0
        })
        return vis, people_data

    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b['bbox']
        area = (x2 - x1) * (y2 - y1)  # Add area to bbox_info
        b['area'] = area

        pad = 40
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(img.shape[1], x2 + pad); y2 = min(img.shape[0], y2 + pad)

        crop = img[y1:y2, x1:x2]
        sk33 = extract_skel_33(crop)
        if sk33 is None:
            print(f"Person {i}: No pose detected")
            continue

        sk25_crop = remap_to_body25(sk33)

        # Transform coordinates back to full image
        sk25 = sk25_crop.copy()
        sk25[:, 0] += x1
        sk25[:, 1] += y1

        draw_skel(vis, sk25)
        
        # Store complete person data
        people_data.append({
            'skeleton': sk25,
            'bbox_info': b,
            'person_id': i
        })

        print(f"Person {i}: pose detected, conf {b['conf']:.2f}, area {area}")

    return vis, people_data


def display_result(image, title="Pose Detection Result"):
    """Display image using matplotlib."""
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    from main_character_selector import select_main_character, visualize_selection
    
    ap = argparse.ArgumentParser(description="Multi-person pose detection with main character selection")
    ap.add_argument("--input", type=str, help="Path to input image (optional, random if not provided)")
    ap.add_argument("--folder", type=str, default="poses/sample_images",
                    help="Folder to pick random image from (default: poses/sample_images)")
    ap.add_argument("--method", type=str, default="weighted",
                    choices=['largest_bbox', 'weighted'],
                    help="Main character selection method: largest_bbox or weighted (bbox size + confidence)")
    args = ap.parse_args()
    
    # Get image path
    if args.input:
        image_path = args.input
    else:
        print(f"No input specified, selecting random image from '{args.folder}'...")
        image_path = get_random_image(args.folder)
    
    # Process and collect all people
    vis_img, people_data = run_on_image(image_path)
    
    if len(people_data) > 0:
        print(f"\nDetected {len(people_data)} person(s)")
        
        # Select main character
        main_idx, main_skeleton, selection_info = select_main_character(people_data, method=args.method)
        
        print(f"\n{'='*50}")
        print(f"MAIN CHARACTER SELECTED:")
        print(f"  Person ID: {selection_info['main_person_id']}")
        print(f"  Method: {selection_info['method']}")
        print(f"  Bbox Area: {selection_info['bbox_area']}")
        print(f"  Confidence: {selection_info['confidence']:.2f}")
        if 'score' in selection_info:
            print(f"  Combined Score: {selection_info['score']:.2f}")
        print(f"{'='*50}\n")
        
        # Visualize selection
        selection_vis = visualize_selection(vis_img, people_data, main_idx)
        display_result(selection_vis, f"Main Character Selection: {Path(image_path).name}")
        
    else:
        print("\nNo people detected")
        display_result(vis_img, f"Pose Detection: {Path(image_path).name}")