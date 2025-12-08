#!/usr/bin/env python3
"""
Edge TPU Client Library

Connect to the Edge TPU inference server from any Python version.
Works with optidex, vr-passthrough, or any other Python application.

Usage:
    from edgetpu_client import EdgeTPUClient
    
    client = EdgeTPUClient()
    
    # Object detection
    with open('image.jpg', 'rb') as f:
        detections = client.detect(f.read())
    
    # Pose estimation
    with open('image.jpg', 'rb') as f:
        pose = client.estimate_pose(f.read())
"""

import socket
import struct
import json
import time
from typing import Dict, Any, Optional, List, Union
from io import BytesIO

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class EdgeTPUClient:
    """Client for Edge TPU inference server."""
    
    def __init__(self, 
                 host: str = 'localhost',
                 detection_port: int = 5590,
                 pose_port: int = 5591,
                 timeout: float = 10.0):
        """
        Initialize Edge TPU client.
        
        Args:
            host: Server hostname (default: localhost)
            detection_port: Detection service port (default: 5590)
            pose_port: Pose estimation service port (default: 5591)
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.detection_port = detection_port
        self.pose_port = pose_port
        self.timeout = timeout
    
    def _send_request(self, port: int, data: bytes) -> Dict[str, Any]:
        """Send request to server and get response."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        
        try:
            sock.connect((self.host, port))
            
            # Send length-prefixed message
            sock.send(struct.pack('>I', len(data)))
            sock.send(data)
            
            # Receive response length
            length_data = sock.recv(4)
            if not length_data:
                raise ConnectionError("Server closed connection")
            
            response_length = struct.unpack('>I', length_data)[0]
            
            # Receive response
            response_data = b''
            while len(response_data) < response_length:
                chunk = sock.recv(min(response_length - len(response_data), 65536))
                if not chunk:
                    break
                response_data += chunk
            
            return json.loads(response_data.decode('utf-8'))
            
        finally:
            sock.close()
    
    def _send_command(self, port: int, cmd: str, **kwargs) -> Dict[str, Any]:
        """Send JSON command to server."""
        request = {'cmd': cmd, **kwargs}
        return self._send_request(port, json.dumps(request).encode('utf-8'))
    
    def ping(self, service: str = 'detection') -> bool:
        """
        Ping the server to check if it's running.
        
        Args:
            service: 'detection' or 'pose'
            
        Returns:
            True if server responds
        """
        port = self.detection_port if service == 'detection' else self.pose_port
        try:
            response = self._send_command(port, 'ping')
            return response.get('success', False)
        except Exception:
            return False
    
    def status(self, service: str = 'detection') -> Dict[str, Any]:
        """
        Get server status.
        
        Args:
            service: 'detection' or 'pose'
            
        Returns:
            Status dict with model info
        """
        port = self.detection_port if service == 'detection' else self.pose_port
        return self._send_command(port, 'status')
    
    def detect(self, 
               image: Union[bytes, 'np.ndarray', 'Image.Image'],
               threshold: float = 0.5) -> Dict[str, Any]:
        """
        Run object detection on image.
        
        Args:
            image: Image as bytes (JPEG/PNG), numpy array, or PIL Image
            threshold: Detection confidence threshold (0.0-1.0)
            
        Returns:
            Dict with:
                - success: bool
                - detections: List of detection dicts
                - count: Number of detections
                - inference_time_ms: Inference time in ms
                - total_time_ms: Total processing time in ms
                - image_size: [width, height]
        """
        image_bytes = self._to_bytes(image)
        return self._send_request(self.detection_port, image_bytes)
    
    def estimate_pose(self,
                      image: Union[bytes, 'np.ndarray', 'Image.Image'],
                      threshold: float = 0.3) -> Dict[str, Any]:
        """
        Run pose estimation on image.
        
        Args:
            image: Image as bytes (JPEG/PNG), numpy array, or PIL Image
            threshold: Keypoint confidence threshold (0.0-1.0)
            
        Returns:
            Dict with:
                - success: bool
                - keypoints: List of keypoint dicts
                - visible_count: Number of visible keypoints
                - inference_time_ms: Inference time in ms
                - total_time_ms: Total processing time in ms
                - image_size: [width, height]
        """
        image_bytes = self._to_bytes(image)
        return self._send_request(self.pose_port, image_bytes)
    
    def _to_bytes(self, image: Union[bytes, 'np.ndarray', 'Image.Image']) -> bytes:
        """Convert image to JPEG bytes."""
        if isinstance(image, bytes):
            return image
        
        if not PIL_AVAILABLE:
            raise ImportError("PIL required for numpy/Image conversion")
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            return buffer.getvalue()
        
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    def detect_objects(self, 
                       image: Union[bytes, 'np.ndarray', 'Image.Image'],
                       classes: Optional[List[str]] = None,
                       threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Convenience method: detect objects and optionally filter by class.
        
        Args:
            image: Input image
            classes: Optional list of class names to filter (e.g., ['person', 'car'])
            threshold: Detection confidence threshold
            
        Returns:
            List of detection dicts
        """
        result = self.detect(image, threshold)
        
        if not result.get('success'):
            return []
        
        detections = result.get('detections', [])
        
        if classes:
            classes_lower = [c.lower() for c in classes]
            detections = [d for d in detections if d['class_name'].lower() in classes_lower]
        
        return detections
    
    def is_person_detected(self, 
                           image: Union[bytes, 'np.ndarray', 'Image.Image'],
                           threshold: float = 0.5) -> bool:
        """Check if a person is detected in the image."""
        detections = self.detect_objects(image, classes=['person'], threshold=threshold)
        return len(detections) > 0
    
    def get_keypoints(self,
                      image: Union[bytes, 'np.ndarray', 'Image.Image'],
                      threshold: float = 0.3) -> Dict[str, Dict[str, Any]]:
        """
        Get pose keypoints as a dict indexed by keypoint name.
        
        Args:
            image: Input image
            threshold: Keypoint confidence threshold
            
        Returns:
            Dict mapping keypoint name to {x, y, confidence, visible}
        """
        result = self.estimate_pose(image, threshold)
        
        if not result.get('success'):
            return {}
        
        return {kp['name']: kp for kp in result.get('keypoints', [])}


class DualTPUClient(EdgeTPUClient):
    """
    Client that can use both TPUs in parallel for stereo vision.
    
    Detection runs on TPU 0, Pose runs on TPU 1.
    """
    
    def detect_stereo(self,
                      left_image: Union[bytes, 'np.ndarray', 'Image.Image'],
                      right_image: Union[bytes, 'np.ndarray', 'Image.Image'],
                      threshold: float = 0.5) -> Dict[str, List[Dict]]:
        """
        Run detection on stereo image pair.
        
        Note: Currently sequential, could be made parallel with threading.
        
        Args:
            left_image: Left eye image
            right_image: Right eye image
            threshold: Detection threshold
            
        Returns:
            Dict with 'left' and 'right' detection lists
        """
        left_result = self.detect(left_image, threshold)
        right_result = self.detect(right_image, threshold)
        
        return {
            'left': left_result.get('detections', []),
            'right': right_result.get('detections', []),
            'left_time_ms': left_result.get('inference_time_ms', 0),
            'right_time_ms': right_result.get('inference_time_ms', 0)
        }


def test_client():
    """Test the Edge TPU client."""
    print("Testing Edge TPU Client...")
    print()
    
    client = EdgeTPUClient()
    
    # Test ping
    print("Pinging detection server...")
    if client.ping('detection'):
        print("  ✓ Detection server is running")
    else:
        print("  ✗ Detection server not available")
        return
    
    print("Pinging pose server...")
    if client.ping('pose'):
        print("  ✓ Pose server is running")
    else:
        print("  ✗ Pose server not available")
    
    # Test status
    print()
    print("Getting server status...")
    status = client.status('detection')
    print(f"  Detection: {status}")
    
    status = client.status('pose')
    print(f"  Pose: {status}")
    
    # Test with a dummy image if PIL is available
    if PIL_AVAILABLE:
        print()
        print("Testing inference with dummy image...")
        
        # Create a dummy 640x480 image
        dummy_image = Image.new('RGB', (640, 480), color='gray')
        
        # Test detection
        start = time.time()
        result = client.detect(dummy_image)
        elapsed = (time.time() - start) * 1000
        print(f"  Detection: {result.get('count', 0)} objects, "
              f"inference: {result.get('inference_time_ms', 0):.1f}ms, "
              f"total: {elapsed:.1f}ms")
        
        # Test pose
        start = time.time()
        result = client.estimate_pose(dummy_image)
        elapsed = (time.time() - start) * 1000
        print(f"  Pose: {result.get('visible_count', 0)} keypoints, "
              f"inference: {result.get('inference_time_ms', 0):.1f}ms, "
              f"total: {elapsed:.1f}ms")
    
    print()
    print("✓ Client test complete!")


if __name__ == '__main__':
    test_client()

