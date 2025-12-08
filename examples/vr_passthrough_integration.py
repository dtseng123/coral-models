#!/usr/bin/env python3
"""
VR Passthrough Integration Example

Shows how to use Edge TPU for real-time hand/pose tracking
in VR passthrough applications.

The dual TPU setup allows parallel processing:
- TPU 0: Object detection (hands, controllers)
- TPU 1: Pose estimation (body tracking)
"""

import sys
sys.path.insert(0, '/home/dash/coral-models')

from edgetpu_client import EdgeTPUClient
import time

# Optional: for camera capture
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class VRVisionProcessor:
    """Vision processor for VR passthrough using dual Edge TPUs."""
    
    def __init__(self, host='localhost'):
        self.client = EdgeTPUClient(host=host)
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Edge TPU server is available."""
        if not self.client.ping('detection'):
            raise ConnectionError("Edge TPU detection server not available on port 5590")
        if not self.client.ping('pose'):
            raise ConnectionError("Edge TPU pose server not available on port 5591")
        print("✓ Connected to Edge TPU servers")
    
    def process_frame(self, frame):
        """
        Process a camera frame for VR passthrough.
        
        Args:
            frame: numpy array (H, W, 3) BGR format from cv2
            
        Returns:
            dict with:
                - people: list of detected people with bboxes
                - hands: estimated hand positions
                - pose: body keypoints if person detected
        """
        # Convert BGR to RGB for the model
        if CV2_AVAILABLE and len(frame.shape) == 3 and frame.shape[2] == 3:
            from PIL import Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
        else:
            image = frame
        
        results = {
            'people': [],
            'hands': [],
            'pose': None
        }
        
        # Run object detection
        detection_result = self.client.detect(image, threshold=0.5)
        
        if detection_result.get('success'):
            for det in detection_result.get('detections', []):
                if det['class_name'] == 'person':
                    results['people'].append({
                        'bbox': det['bbox'],
                        'confidence': det['confidence']
                    })
        
        # Run pose estimation (only if person detected for efficiency)
        if results['people']:
            pose_result = self.client.estimate_pose(image, threshold=0.3)
            
            if pose_result.get('success'):
                keypoints = {kp['name']: kp for kp in pose_result.get('keypoints', [])}
                results['pose'] = keypoints
                
                # Extract hand positions
                for hand in ['left_wrist', 'right_wrist']:
                    if hand in keypoints and keypoints[hand]['visible']:
                        results['hands'].append({
                            'name': hand.replace('_wrist', ''),
                            'x': keypoints[hand]['x'],
                            'y': keypoints[hand]['y'],
                            'confidence': keypoints[hand]['confidence']
                        })
        
        return results
    
    def track_hands(self, frame):
        """
        Quick hand tracking for controller-free VR.
        
        Args:
            frame: Camera frame
            
        Returns:
            List of hand positions [(x, y, confidence), ...]
        """
        results = self.process_frame(frame)
        return [(h['x'], h['y'], h['confidence']) for h in results['hands']]


def demo_with_camera():
    """Demo with USB camera."""
    if not CV2_AVAILABLE:
        print("OpenCV not available for camera demo")
        return
    
    processor = VRVisionProcessor()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit")
    
    fps_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        results = processor.process_frame(frame)
        process_time = (time.time() - start) * 1000
        fps_times.append(process_time)
        
        # Draw results
        for person in results['people']:
            bbox = person['bbox']
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        for hand in results['hands']:
            cv2.circle(frame, (hand['x'], hand['y']), 10, (0, 0, 255), -1)
            cv2.putText(frame, hand['name'], (hand['x']+15, hand['y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS display
        if len(fps_times) > 10:
            fps_times.pop(0)
        avg_ms = sum(fps_times) / len(fps_times)
        cv2.putText(frame, f"{1000/avg_ms:.0f} FPS ({avg_ms:.0f}ms)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('VR Passthrough Vision', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("VR Passthrough Vision Demo")
    print("=" * 40)
    
    # Quick test
    processor = VRVisionProcessor()
    
    if CV2_AVAILABLE:
        print("\nStarting camera demo...")
        demo_with_camera()
    else:
        print("\nInstall opencv-python for camera demo:")
        print("  pip install opencv-python")

