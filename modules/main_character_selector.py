"""
main_character_selector.py

Main character selection from multiple detected poses:
1. Largest bounding box method
2. Weighted scoring (bbox size + detection confidence)
3. Visualization helpers
"""

import numpy as np
import cv2

def select_main_character(people_data, method='largest_bbox', weights=None):
    """
    Select the main character from detected people using specified method.
    
    Args:
        people_data: List of dictionaries with keys:
            - skeleton: np.array(25, 3) BODY25 format skeleton
            - bbox_info: dict with 'bbox', 'conf', 'area' keys
            - person_id: int, detection index
        method: 'largest_bbox' or 'weighted'
        weights: Optional dict with 'area': float, 'conf': float
                for weighted scoring (default: {'area': 0.7, 'conf': 0.3})
    
    Returns:
        (index, skeleton, selection_info) tuple
    """
    if not people_data:
        raise ValueError("No people detected in the image")
    
    if method == 'largest_bbox':
        # Simply pick person with largest bounding box
        areas = [p['bbox_info']['area'] for p in people_data]
        main_idx = np.argmax(areas)
        
        return main_idx, people_data[main_idx]['skeleton'], {
            'method': 'largest_bbox',
            'main_person_id': people_data[main_idx]['person_id'],
            'bbox_area': people_data[main_idx]['bbox_info']['area'],
            'confidence': people_data[main_idx]['bbox_info']['conf']
        }
    
    elif method == 'weighted':
        if weights is None:
            weights = {'area': 0.7, 'conf': 0.3}
        
        # Normalize areas and confidences to [0, 1]
        areas = np.array([p['bbox_info']['area'] for p in people_data])
        confs = np.array([p['bbox_info']['conf'] for p in people_data])
        
        areas = (areas - areas.min()) / (areas.max() - areas.min() + 1e-6)
        
        # Weighted sum score
        scores = weights['area'] * areas + weights['conf'] * confs
        main_idx = np.argmax(scores)
        
        return main_idx, people_data[main_idx]['skeleton'], {
            'method': 'weighted',
            'main_person_id': people_data[main_idx]['person_id'],
            'bbox_area': people_data[main_idx]['bbox_info']['area'],
            'confidence': people_data[main_idx]['bbox_info']['conf'],
            'score': float(scores[main_idx])
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")


def visualize_selection(image, people_data, main_idx):
    """
    Draw all detected skeletons but highlight the main character.
    
    Args:
        image: RGB/BGR image to draw on
        people_data: List of person dictionaries
        main_idx: Index of the main character in people_data
    
    Returns:
        Annotated image copy
    """
    vis = image.copy()
    
    # First draw all other skeletons in gray
    for i, person in enumerate(people_data):
        if i == main_idx:
            continue
            
        sk = person['skeleton']
        x1, y1, x2, y2 = person['bbox_info']['bbox']
        
        # Draw grayed-out skeleton
        for j in range(len(sk)):
            if sk[j, 2] > 0.1:  # visibility check
                cv2.circle(vis, (int(sk[j, 0]), int(sk[j, 1])), 3, (128, 128, 128), -1)
        
        # Gray bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 128, 128), 2)
    
    # Then draw main character in color
    main_person = people_data[main_idx]
    sk = main_person['skeleton']
    x1, y1, x2, y2 = main_person['bbox_info']['bbox']
    
    # Highlighted skeleton
    for j in range(len(sk)):
        if sk[j, 2] > 0.1:
            cv2.circle(vis, (int(sk[j, 0]), int(sk[j, 1])), 4, (0, 0, 255), -1)
    
    # Green bounding box
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add confidence score
    conf = main_person['bbox_info']['conf']
    cv2.putText(vis, f"Main: {conf:.2f}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return vis