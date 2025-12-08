#!/usr/bin/env python3
"""
Edge TPU Inference Server for Raspberry Pi 5
Runs in Docker with pycoral for full TPU support.

Provides socket-based API for:
- Object detection (SSD MobileNet, EfficientDet)
- Pose estimation (MoveNet)

Usage:
    docker run -d --name edgetpu-server \
        --device /dev/apex_0 --device /dev/apex_1 \
        -p 5590:5590 -p 5591:5591 \
        -v /path/to/models:/app/models \
        edgetpu-server
"""

import socket
import json
import threading
import time
import os
import sys
import struct
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from io import BytesIO

# Import pycoral
try:
    from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter
    from pycoral.adapters import common, detect
    from PIL import Image
    PYCORAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pycoral not available: {e}")
    PYCORAL_AVAILABLE = False

# Configuration
DETECTION_PORT = 5590
POSE_PORT = 5591
MODEL_DIR = "/app/models"

# Model paths
MODELS = {
    'ssd_mobilenet': 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
    'efficientdet': 'efficientdet_lite0_edgetpu.tflite',
    'movenet': 'movenet_lightning_edgetpu.tflite',
}

# COCO labels
COCO_LABELS = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench',
    15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep',
    20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe',
    26: 'backpack', 27: 'umbrella', 30: 'handbag', 31: 'tie', 32: 'suitcase',
    33: 'frisbee', 34: 'skis', 35: 'snowboard', 36: 'sports ball', 37: 'kite',
    38: 'baseball bat', 39: 'baseball glove', 40: 'skateboard', 41: 'surfboard',
    42: 'tennis racket', 43: 'bottle', 45: 'wine glass', 46: 'cup',
    47: 'fork', 48: 'knife', 49: 'spoon', 50: 'bowl', 51: 'banana',
    52: 'apple', 53: 'sandwich', 54: 'orange', 55: 'broccoli', 56: 'carrot',
    57: 'hot dog', 58: 'pizza', 59: 'donut', 60: 'cake', 61: 'chair',
    62: 'couch', 63: 'potted plant', 64: 'bed', 66: 'dining table',
    69: 'toilet', 71: 'tv', 72: 'laptop', 73: 'mouse', 74: 'remote',
    75: 'keyboard', 76: 'cell phone', 77: 'microwave', 78: 'oven',
    79: 'toaster', 80: 'sink', 81: 'refrigerator', 83: 'book', 84: 'clock',
    85: 'vase', 86: 'scissors', 87: 'teddy bear', 88: 'hair drier', 89: 'toothbrush',
}

# MoveNet keypoint names
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


class EdgeTPUDetector:
    """Object detection using Edge TPU."""
    
    def __init__(self, model_name: str = 'ssd_mobilenet', device: int = 0):
        self.model_name = model_name
        self.device = device
        self.interpreter = None
        self.input_size = None
        
        model_path = os.path.join(MODEL_DIR, MODELS.get(model_name, model_name))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create interpreter - let pycoral auto-select device for first model
        # For subsequent models, we could specify device but auto works well
        print(f"Loading {model_name}...")
        
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        
        # Get input size
        input_details = self.interpreter.get_input_details()[0]
        self.input_size = (input_details['shape'][2], input_details['shape'][1])  # (width, height)
        
        print(f"  Model loaded! Input size: {self.input_size}")
    
    def detect(self, image_data: bytes, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run object detection on image.
        
        Args:
            image_data: JPEG or PNG image bytes
            threshold: Detection confidence threshold
            
        Returns:
            Dict with detections and timing info
        """
        start_time = time.time()
        
        # Decode image
        image = Image.open(BytesIO(image_data)).convert('RGB')
        orig_width, orig_height = image.size
        
        # Resize to model input size
        image_resized = image.resize(self.input_size, Image.BILINEAR)
        
        # Set input tensor
        common.set_input(self.interpreter, image_resized)
        
        # Run inference
        inference_start = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - inference_start) * 1000
        
        # Get detections
        detections = detect.get_objects(self.interpreter, threshold)
        
        # Format results
        results = []
        for det in detections:
            # Scale bbox to original image size
            bbox = det.bbox
            results.append({
                'class_id': det.id,
                'class_name': COCO_LABELS.get(det.id, f'class_{det.id}'),
                'confidence': float(det.score),
                'bbox': [
                    int(bbox.xmin * orig_width / self.input_size[0]),
                    int(bbox.ymin * orig_height / self.input_size[1]),
                    int(bbox.xmax * orig_width / self.input_size[0]),
                    int(bbox.ymax * orig_height / self.input_size[1])
                ]
            })
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'success': True,
            'detections': results,
            'count': len(results),
            'inference_time_ms': round(inference_time, 1),
            'total_time_ms': round(total_time, 1),
            'image_size': [orig_width, orig_height]
        }


class EdgeTPUPoseEstimator:
    """Pose estimation using MoveNet on Edge TPU."""
    
    def __init__(self, device: int = 0):
        self.device = device
        self.interpreter = None
        self.input_size = None
        
        model_path = os.path.join(MODEL_DIR, MODELS['movenet'])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MoveNet model not found: {model_path}")
        
        print(f"Loading MoveNet...")
        
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        
        input_details = self.interpreter.get_input_details()[0]
        self.input_size = (input_details['shape'][2], input_details['shape'][1])
        
        print(f"  MoveNet loaded! Input size: {self.input_size}")
    
    def estimate(self, image_data: bytes, threshold: float = 0.3) -> Dict[str, Any]:
        """
        Estimate pose keypoints.
        
        Args:
            image_data: JPEG or PNG image bytes
            threshold: Keypoint confidence threshold
            
        Returns:
            Dict with keypoints and timing info
        """
        start_time = time.time()
        
        # Decode image
        image = Image.open(BytesIO(image_data)).convert('RGB')
        orig_width, orig_height = image.size
        
        # Resize to model input size
        image_resized = image.resize(self.input_size, Image.BILINEAR)
        
        # Set input tensor
        common.set_input(self.interpreter, image_resized)
        
        # Run inference
        inference_start = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - inference_start) * 1000
        
        # Get output (shape: [1, 1, 17, 3] for MoveNet)
        output = self.interpreter.get_output_details()[0]
        keypoints_raw = self.interpreter.tensor(output['index'])()[0, 0]
        
        # Format keypoints
        keypoints = []
        for i, (y, x, conf) in enumerate(keypoints_raw):
            keypoints.append({
                'name': KEYPOINT_NAMES[i],
                'x': int(float(x) * orig_width),
                'y': int(float(y) * orig_height),
                'confidence': float(conf),
                'visible': bool(float(conf) >= threshold)
            })
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'success': True,
            'keypoints': keypoints,
            'visible_count': sum(1 for kp in keypoints if kp['visible']),
            'inference_time_ms': round(inference_time, 1),
            'total_time_ms': round(total_time, 1),
            'image_size': [orig_width, orig_height]
        }


class InferenceServer:
    """Socket server for Edge TPU inference requests."""
    
    def __init__(self, port: int, service_type: str):
        self.port = port
        self.service_type = service_type
        self.running = False
        self.server_socket = None
        
        # Initialize the appropriate model
        if service_type == 'detection':
            # Use TPU 0 for detection
            self.model = EdgeTPUDetector(model_name='ssd_mobilenet', device=0)
        elif service_type == 'pose':
            # Use TPU 1 for pose (or TPU 0 if only one available)
            try:
                self.model = EdgeTPUPoseEstimator(device=1)
            except Exception:
                print("TPU 1 not available, using TPU 0 for pose")
                self.model = EdgeTPUPoseEstimator(device=0)
    
    def handle_client(self, client_socket: socket.socket, address: Tuple):
        """Handle a client connection."""
        print(f"[{self.service_type}] Client connected: {address}")
        
        try:
            while self.running:
                # Read message length (4 bytes, big-endian)
                length_data = client_socket.recv(4)
                if not length_data:
                    break
                
                msg_length = struct.unpack('>I', length_data)[0]
                
                # Read message
                data = b''
                while len(data) < msg_length:
                    chunk = client_socket.recv(min(msg_length - len(data), 65536))
                    if not chunk:
                        break
                    data += chunk
                
                if len(data) < msg_length:
                    break
                
                # Parse request
                try:
                    # First try JSON command
                    request = json.loads(data.decode('utf-8'))
                    cmd = request.get('cmd', 'detect')
                    
                    if cmd == 'ping':
                        response = {'success': True, 'message': 'pong', 'service': self.service_type}
                    elif cmd == 'status':
                        response = {
                            'success': True,
                            'service': self.service_type,
                            'model': self.model.model_name if hasattr(self.model, 'model_name') else 'movenet',
                            'input_size': list(self.model.input_size)
                        }
                    else:
                        response = {'error': f'Unknown command: {cmd}'}
                    
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Binary data - assume it's an image
                    threshold = 0.5  # Default threshold
                    
                    if self.service_type == 'detection':
                        response = self.model.detect(data, threshold)
                    else:
                        response = self.model.estimate(data, threshold)
                
                # Send response
                response_json = json.dumps(response).encode('utf-8')
                client_socket.send(struct.pack('>I', len(response_json)))
                client_socket.send(response_json)
                
        except Exception as e:
            print(f"[{self.service_type}] Client error: {e}")
        finally:
            client_socket.close()
            print(f"[{self.service_type}] Client disconnected: {address}")
    
    def start(self):
        """Start the inference server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        
        self.running = True
        print(f"[{self.service_type}] Server listening on port {self.port}")
        
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[{self.service_type}] Accept error: {e}")
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Edge TPU Inference Server")
    print("=" * 60)
    
    if not PYCORAL_AVAILABLE:
        print("ERROR: pycoral not available!")
        sys.exit(1)
    
    # List available TPUs
    tpus = list_edge_tpus()
    print(f"Found {len(tpus)} Edge TPU(s):")
    for tpu in tpus:
        print(f"  - {tpu['type']}: {tpu['path']}")
    
    if not tpus:
        print("ERROR: No Edge TPUs found!")
        sys.exit(1)
    
    print()
    
    # Check models exist
    print("Checking models...")
    for name, filename in MODELS.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            print(f"  ✓ {name}: {filename}")
        else:
            print(f"  ✗ {name}: {filename} (NOT FOUND)")
    
    print()
    
    # Start servers
    servers = []
    
    try:
        # Detection server on TPU 0
        print("Starting detection server...")
        detection_server = InferenceServer(DETECTION_PORT, 'detection')
        detection_thread = threading.Thread(target=detection_server.start, daemon=True)
        detection_thread.start()
        servers.append(detection_server)
        
        # Pose server on TPU 1 (or TPU 0 if only one)
        print("Starting pose server...")
        pose_server = InferenceServer(POSE_PORT, 'pose')
        pose_thread = threading.Thread(target=pose_server.start, daemon=True)
        pose_thread.start()
        servers.append(pose_server)
        
        print()
        print("=" * 60)
        print("Servers running:")
        print(f"  Detection: port {DETECTION_PORT}")
        print(f"  Pose:      port {POSE_PORT}")
        print("=" * 60)
        print("Press Ctrl+C to stop")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        for server in servers:
            server.stop()


if __name__ == '__main__':
    main()

