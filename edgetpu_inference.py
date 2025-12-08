#!/usr/bin/env python3
"""
Edge TPU Inference Wrapper for Dual Coral TPU on Raspberry Pi 5

Supports:
- Object detection (SSD MobileNet, EfficientDet)
- Pose estimation (MoveNet)
- Parallel inference on dual TPUs

Usage:
    from edgetpu_inference import EdgeTPUDetector, EdgeTPUPoseEstimator
    
    # Object detection
    detector = EdgeTPUDetector(device=0)  # Use TPU 0
    detections = detector.detect(image)
    
    # Pose estimation
    pose = EdgeTPUPoseEstimator(device=1)  # Use TPU 1
    keypoints = pose.estimate(image)
"""

import os
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Try to import TFLite runtime (ai-edge-litert is the new package)
try:
    from ai_edge_litert.interpreter import Interpreter, load_delegate
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
        load_delegate = tflite.load_delegate
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("Warning: TFLite runtime not available")

# Model paths
MODEL_DIR = Path(__file__).parent
MODELS = {
    'ssd_mobilenet': MODEL_DIR / 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
    'efficientdet': MODEL_DIR / 'efficientdet_lite0_edgetpu.tflite',
    'movenet': MODEL_DIR / 'movenet_lightning_edgetpu.tflite',
}

# COCO labels for detection models
COCO_LABELS = MODEL_DIR / 'coco_labels.txt'

# Edge TPU library path
EDGETPU_LIB = '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1'


def load_labels(path: Path) -> Dict[int, str]:
    """Load label map from file."""
    labels = {}
    if path.exists():
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                labels[i] = line.strip()
    return labels


class EdgeTPUBase:
    """Base class for Edge TPU inference."""
    
    def __init__(self, model_path: str, device: int = 0):
        """
        Initialize Edge TPU interpreter.
        
        Args:
            model_path: Path to EdgeTPU-compiled TFLite model
            device: TPU device index (0 or 1 for dual TPU)
        """
        self.model_path = model_path
        self.device = device
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._delegate = None
        
        if not TFLITE_AVAILABLE:
            raise RuntimeError("TFLite runtime not available")
        
        self._load_model()
    
    def _load_model(self):
        """Load model with Edge TPU delegate."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Try to load with Edge TPU delegate
        try:
            device_str = f':{self.device}' if self.device > 0 else ''
            self._delegate = load_delegate(
                EDGETPU_LIB, 
                {'device': f'pci{device_str}'}
            )
            self.interpreter = Interpreter(
                model_path=str(self.model_path),
                experimental_delegates=[self._delegate]
            )
            print(f"Loaded model on Edge TPU {self.device}")
        except Exception as e:
            print(f"Warning: Could not load Edge TPU delegate: {e}")
            print("Falling back to CPU inference")
            self.interpreter = Interpreter(model_path=str(self.model_path))
        
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        # Get expected input shape
        input_shape = self.input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        
        # Resize image
        from PIL import Image
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        img = img.resize((width, height), Image.BILINEAR)
        
        # Convert to numpy and add batch dimension
        input_data = np.array(img, dtype=np.uint8)
        if len(input_data.shape) == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
    
    def _invoke(self, input_data: np.ndarray) -> float:
        """Run inference and return elapsed time in ms."""
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        start = time.perf_counter()
        self.interpreter.invoke()
        elapsed = (time.perf_counter() - start) * 1000
        
        return elapsed


class EdgeTPUDetector(EdgeTPUBase):
    """Object detection using Edge TPU."""
    
    def __init__(self, model: str = 'ssd_mobilenet', device: int = 0, 
                 confidence_threshold: float = 0.5):
        """
        Initialize object detector.
        
        Args:
            model: Model name ('ssd_mobilenet' or 'efficientdet')
            device: TPU device index
            confidence_threshold: Minimum confidence for detections
        """
        model_path = MODELS.get(model)
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(f"Model '{model}' not found")
        
        super().__init__(str(model_path), device)
        self.confidence_threshold = confidence_threshold
        self.labels = load_labels(COCO_LABELS)
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in image.
        
        Args:
            image: Input image (numpy array, RGB)
            
        Returns:
            List of detections, each with:
                - bbox: [x1, y1, x2, y2] normalized coordinates
                - class_id: Class index
                - class_name: Class label
                - confidence: Detection confidence
        """
        # Get original image size for scaling boxes
        if isinstance(image, np.ndarray):
            orig_height, orig_width = image.shape[:2]
        else:
            orig_width, orig_height = image.size
        
        # Preprocess and run inference
        input_data = self._preprocess(image)
        inference_time = self._invoke(input_data)
        
        # Parse outputs (format depends on model)
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        # Filter and format detections
        detections = []
        for i in range(len(scores)):
            if scores[i] >= self.confidence_threshold:
                # Box format: [y1, x1, y2, x2] normalized
                y1, x1, y2, x2 = boxes[i]
                
                # Scale to original image coordinates
                bbox = [
                    int(x1 * orig_width),
                    int(y1 * orig_height),
                    int(x2 * orig_width),
                    int(y2 * orig_height)
                ]
                
                class_id = int(classes[i])
                detections.append({
                    'bbox': bbox,
                    'class_id': class_id,
                    'class_name': self.labels.get(class_id, f'class_{class_id}'),
                    'confidence': float(scores[i])
                })
        
        return detections
    
    def detect_with_timing(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """Detect objects and return inference time."""
        input_data = self._preprocess(image)
        inference_time = self._invoke(input_data)
        
        # Get original size
        if isinstance(image, np.ndarray):
            orig_height, orig_width = image.shape[:2]
        else:
            orig_width, orig_height = image.size
        
        # Parse outputs
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        detections = []
        for i in range(len(scores)):
            if scores[i] >= self.confidence_threshold:
                y1, x1, y2, x2 = boxes[i]
                bbox = [
                    int(x1 * orig_width), int(y1 * orig_height),
                    int(x2 * orig_width), int(y2 * orig_height)
                ]
                class_id = int(classes[i])
                detections.append({
                    'bbox': bbox,
                    'class_id': class_id,
                    'class_name': self.labels.get(class_id, f'class_{class_id}'),
                    'confidence': float(scores[i])
                })
        
        return detections, inference_time


class EdgeTPUPoseEstimator(EdgeTPUBase):
    """Pose estimation using MoveNet on Edge TPU."""
    
    # MoveNet keypoint indices
    KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, device: int = 0, confidence_threshold: float = 0.3):
        """
        Initialize pose estimator.
        
        Args:
            device: TPU device index
            confidence_threshold: Minimum confidence for keypoints
        """
        model_path = MODELS.get('movenet')
        if model_path is None or not model_path.exists():
            raise FileNotFoundError("MoveNet model not found")
        
        super().__init__(str(model_path), device)
        self.confidence_threshold = confidence_threshold
    
    def estimate(self, image: np.ndarray) -> Dict:
        """
        Estimate pose keypoints.
        
        Args:
            image: Input image (numpy array, RGB)
            
        Returns:
            Dict with:
                - keypoints: List of (x, y, confidence) for each keypoint
                - keypoint_names: List of keypoint names
        """
        # Preprocess
        input_data = self._preprocess(image)
        
        # Get original size for scaling
        if isinstance(image, np.ndarray):
            orig_height, orig_width = image.shape[:2]
        else:
            orig_width, orig_height = image.size
        
        # Run inference
        inference_time = self._invoke(input_data)
        
        # Parse output - shape is [1, 1, 17, 3] for MoveNet
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        keypoints_raw = output[0, 0]  # Shape: [17, 3]
        
        # Format keypoints
        keypoints = []
        for i, (y, x, conf) in enumerate(keypoints_raw):
            keypoints.append({
                'name': self.KEYPOINTS[i],
                'x': int(x * orig_width),
                'y': int(y * orig_height),
                'confidence': float(conf),
                'visible': conf >= self.confidence_threshold
            })
        
        return {
            'keypoints': keypoints,
            'inference_time_ms': inference_time
        }


class DualTPUDetector:
    """Parallel detection on dual Edge TPUs for stereo vision."""
    
    def __init__(self, model: str = 'ssd_mobilenet', confidence_threshold: float = 0.5):
        """
        Initialize dual TPU detector.
        
        Args:
            model: Model name
            confidence_threshold: Minimum confidence for detections
        """
        self.detector_left = EdgeTPUDetector(model, device=0, 
                                              confidence_threshold=confidence_threshold)
        self.detector_right = EdgeTPUDetector(model, device=1,
                                               confidence_threshold=confidence_threshold)
    
    def detect_stereo(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[List, List]:
        """
        Detect objects in stereo image pair.
        
        Note: For true parallel execution, you'd want to use threading.
        This is a simple sequential implementation.
        
        Args:
            left_image: Left eye image
            right_image: Right eye image
            
        Returns:
            Tuple of (left_detections, right_detections)
        """
        left_detections = self.detector_left.detect(left_image)
        right_detections = self.detector_right.detect(right_image)
        return left_detections, right_detections


def test_edgetpu():
    """Test Edge TPU functionality."""
    print("Testing Edge TPU inference...")
    print(f"TFLite available: {TFLITE_AVAILABLE}")
    print(f"Models directory: {MODEL_DIR}")
    print()
    
    # Check available models
    print("Available models:")
    for name, path in MODELS.items():
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name}: {path}")
    print()
    
    # Try to create a detector
    try:
        detector = EdgeTPUDetector(model='ssd_mobilenet', device=0)
        print("✓ Detector initialized successfully")
        
        # Test with a dummy image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections, time_ms = detector.detect_with_timing(test_image)
        print(f"✓ Inference completed in {time_ms:.1f}ms")
        print(f"  Found {len(detections)} detections")
        
    except Exception as e:
        print(f"✗ Detector failed: {e}")
    
    print()
    
    # Try pose estimator
    try:
        pose = EdgeTPUPoseEstimator(device=0)
        print("✓ Pose estimator initialized successfully")
        
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = pose.estimate(test_image)
        print(f"✓ Pose estimation completed in {result['inference_time_ms']:.1f}ms")
        print(f"  Detected {len([k for k in result['keypoints'] if k['visible']])} visible keypoints")
        
    except Exception as e:
        print(f"✗ Pose estimator failed: {e}")


if __name__ == '__main__':
    test_edgetpu()





