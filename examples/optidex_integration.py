#!/usr/bin/env python3
"""
Optidex Integration Example

Shows how to integrate Edge TPU inference with optidex
for real-time object detection and tracking.

Features:
- Fast object detection (21ms / 47 FPS)
- Parallel pose estimation (12ms / 82 FPS)
- Socket-based API for any Python version
"""

import sys
sys.path.insert(0, '/home/dash/coral-models')

from edgetpu_client import EdgeTPUClient, DualTPUClient
from typing import List, Dict, Optional
import time


class OptidexVision:
    """Vision module for optidex using Edge TPU."""
    
    # COCO classes relevant for optidex
    RELEVANT_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
        'cat', 'dog', 'chair', 'couch', 'tv', 'laptop', 
        'cell phone', 'bottle', 'cup', 'book'
    ]
    
    def __init__(self, host: str = 'localhost'):
        self.client = EdgeTPUClient(host=host)
        self._connected = False
        self._connect()
    
    def _connect(self):
        """Connect to Edge TPU server."""
        if self.client.ping('detection'):
            self._connected = True
            print("✓ Connected to Edge TPU server")
        else:
            raise ConnectionError("Edge TPU server not available")
    
    @property
    def is_available(self) -> bool:
        """Check if Edge TPU is available."""
        return self._connected and self.client.ping('detection')
    
    def detect(self, 
               image, 
               classes: Optional[List[str]] = None,
               threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in image.
        
        Args:
            image: Image bytes, numpy array, or PIL Image
            classes: Optional filter for specific classes
            threshold: Detection confidence threshold
            
        Returns:
            List of detections with class_name, bbox, confidence
        """
        if classes is None:
            classes = self.RELEVANT_CLASSES
        
        return self.client.detect_objects(image, classes=classes, threshold=threshold)
    
    def find_people(self, image, threshold: float = 0.5) -> List[Dict]:
        """Find all people in image."""
        return self.client.detect_objects(image, classes=['person'], threshold=threshold)
    
    def find_objects(self, image, object_types: List[str], threshold: float = 0.5) -> List[Dict]:
        """Find specific types of objects."""
        return self.client.detect_objects(image, classes=object_types, threshold=threshold)
    
    def get_pose(self, image, threshold: float = 0.3) -> Dict:
        """Get pose keypoints for person in image."""
        return self.client.get_keypoints(image, threshold=threshold)
    
    def process_stereo(self, 
                       left_image, 
                       right_image,
                       classes: Optional[List[str]] = None) -> Dict:
        """
        Process stereo image pair for 3D detection.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            classes: Optional class filter
            
        Returns:
            Dict with left/right detections
        """
        dual_client = DualTPUClient(host=self.client.host)
        result = dual_client.detect_stereo(left_image, right_image)
        
        if classes:
            classes_lower = [c.lower() for c in classes]
            result['left'] = [d for d in result['left'] if d['class_name'].lower() in classes_lower]
            result['right'] = [d for d in result['right'] if d['class_name'].lower() in classes_lower]
        
        return result


class ObjectTracker:
    """Simple object tracker using Edge TPU detections."""
    
    def __init__(self, vision: OptidexVision):
        self.vision = vision
        self.tracked_objects = {}
        self.next_id = 0
    
    def update(self, image) -> List[Dict]:
        """
        Update tracked objects with new frame.
        
        Returns list of tracked objects with IDs.
        """
        detections = self.vision.detect(image)
        
        # Simple IoU-based tracking
        tracked = []
        used_detections = set()
        
        for obj_id, obj in list(self.tracked_objects.items()):
            best_match = None
            best_iou = 0.3  # Min IoU threshold
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                if det['class_name'] != obj['class_name']:
                    continue
                
                iou = self._calculate_iou(obj['bbox'], det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
            
            if best_match is not None:
                det = detections[best_match]
                obj['bbox'] = det['bbox']
                obj['confidence'] = det['confidence']
                obj['frames_seen'] += 1
                used_detections.add(best_match)
                tracked.append({'id': obj_id, **obj})
            else:
                obj['frames_missed'] += 1
                if obj['frames_missed'] > 10:
                    del self.tracked_objects[obj_id]
        
        # Add new detections
        for i, det in enumerate(detections):
            if i not in used_detections:
                obj_id = self.next_id
                self.next_id += 1
                self.tracked_objects[obj_id] = {
                    'class_name': det['class_name'],
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'frames_seen': 1,
                    'frames_missed': 0
                }
                tracked.append({'id': obj_id, **self.tracked_objects[obj_id]})
        
        return tracked
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two bboxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def benchmark():
    """Benchmark the Edge TPU integration."""
    print("=" * 50)
    print("Optidex Vision Benchmark")
    print("=" * 50)
    
    vision = OptidexVision()
    
    # Create test image
    from PIL import Image
    test_image = Image.new('RGB', (640, 480), color='gray')
    
    # Warm up
    for _ in range(5):
        vision.detect(test_image)
    
    # Benchmark detection
    times = []
    for _ in range(50):
        start = time.time()
        vision.detect(test_image)
        times.append((time.time() - start) * 1000)
    
    avg = sum(times) / len(times)
    print(f"\nDetection Performance:")
    print(f"  Average: {avg:.1f}ms")
    print(f"  Min: {min(times):.1f}ms")
    print(f"  Max: {max(times):.1f}ms")
    print(f"  FPS: {1000/avg:.0f}")
    
    # Benchmark pose
    times = []
    for _ in range(50):
        start = time.time()
        vision.get_pose(test_image)
        times.append((time.time() - start) * 1000)
    
    avg = sum(times) / len(times)
    print(f"\nPose Estimation Performance:")
    print(f"  Average: {avg:.1f}ms")
    print(f"  Min: {min(times):.1f}ms")
    print(f"  Max: {max(times):.1f}ms")
    print(f"  FPS: {1000/avg:.0f}")


if __name__ == '__main__':
    benchmark()

