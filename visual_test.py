#!/usr/bin/env python3
"""
Visual Edge TPU Test - Shows camera feed with detection/pose overlays
Supports EMA smoothing and Kalman filter for stable keypoint tracking!
"""

import sys
sys.path.insert(0, '/home/dash/coral-models')

import cv2
import numpy as np
import time
import argparse
from edgetpu_client import EdgeTPUClient
from collections import defaultdict

# Keypoint connections for skeleton drawing
SKELETON_CONNECTIONS = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle'),
    ('left_eye', 'right_eye'),
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_ear', 'left_eye'),
    ('right_ear', 'right_eye'),
]

# Keypoint names in order
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Colors - gradient from head to feet
KEYPOINT_COLORS = {
    'nose': (255, 100, 100),
    'left_eye': (255, 150, 100),
    'right_eye': (255, 150, 100),
    'left_ear': (255, 200, 100),
    'right_ear': (255, 200, 100),
    'left_shoulder': (100, 255, 100),
    'right_shoulder': (100, 255, 100),
    'left_elbow': (100, 255, 200),
    'right_elbow': (100, 255, 200),
    'left_wrist': (100, 255, 255),
    'right_wrist': (100, 255, 255),
    'left_hip': (100, 200, 255),
    'right_hip': (100, 200, 255),
    'left_knee': (100, 100, 255),
    'right_knee': (100, 100, 255),
    'left_ankle': (200, 100, 255),
    'right_ankle': (200, 100, 255),
}

COLORS = {
    'person': (0, 255, 0),
    'default': (255, 200, 0),
    'skeleton': (0, 255, 200),
}


class KeypointKalmanFilter:
    """
    Kalman filter for single 2D keypoint tracking.
    
    State: [x, y, vx, vy] (position + velocity)
    Measurement: [x, y] (position only)
    
    The filter predicts motion and corrects with measurements,
    optimally balancing based on noise estimates.
    """
    
    def __init__(self, process_noise=0.1, measurement_noise=2.0):
        """
        Args:
            process_noise: How much we expect the motion to vary (higher = trust prediction less)
            measurement_noise: How noisy the measurements are (higher = trust measurements less)
        """
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # State covariance (uncertainty)
        self.P = np.eye(4) * 100  # High initial uncertainty
        
        # State transition matrix (constant velocity model)
        # x' = x + vx*dt, y' = y + vy*dt, vx' = vx, vy' = vy
        self.dt = 1.0  # Assume 1 frame timestep
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position, not velocity)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        self.Q[2, 2] *= 2  # Velocity has more uncertainty
        self.Q[3, 3] *= 2
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        self.initialized = False
        self.frames_since_measurement = 0
    
    def predict(self):
        """Predict next state based on motion model."""
        # State prediction
        self.state = self.F @ self.state
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.frames_since_measurement += 1
        
        return self.state[:2]  # Return predicted position
    
    def update(self, measurement, confidence=1.0):
        """
        Update state with new measurement.
        
        Args:
            measurement: [x, y] position
            confidence: Detection confidence (0-1), affects measurement noise
        """
        if not self.initialized:
            # Initialize state with first measurement
            self.state = np.array([measurement[0], measurement[1], 0, 0])
            self.P = np.eye(4) * 10
            self.initialized = True
            self.frames_since_measurement = 0
            return self.state[:2]
        
        # Adjust measurement noise based on confidence
        # Lower confidence = higher noise = trust measurement less
        conf_factor = max(0.1, confidence) ** 2
        R_adjusted = self.R / conf_factor
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + R_adjusted
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Measurement residual
        z = np.array(measurement)
        y = z - self.H @ self.state
        
        # State update
        self.state = self.state + K @ y
        
        # Covariance update
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        self.frames_since_measurement = 0
        
        return self.state[:2]
    
    def get_position(self):
        """Get current estimated position."""
        return self.state[:2]
    
    def get_velocity(self):
        """Get current estimated velocity."""
        return self.state[2:4]
    
    def reset(self):
        """Reset filter state."""
        self.state = np.zeros(4)
        self.P = np.eye(4) * 100
        self.initialized = False
        self.frames_since_measurement = 0


class KalmanKeypointSmoother:
    """
    Kalman filter-based keypoint smoother for stable pose visualization.
    
    Features:
    - Individual Kalman filter per keypoint
    - Confidence-weighted measurement updates
    - Velocity prediction through occlusions
    - Adaptive process noise based on body part
    """
    
    def __init__(self, 
                 process_noise=0.05,
                 measurement_noise=3.0,
                 min_confidence=0.15,
                 display_confidence=0.2,
                 persistence_frames=8):
        
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.min_confidence = min_confidence
        self.display_confidence = display_confidence
        self.persistence_frames = persistence_frames
        
        # Create a Kalman filter for each keypoint
        self.filters = {}
        self.confidences = {}
        
        # Different noise levels for different body parts
        # Extremities (hands, feet) move more = higher process noise
        self.noise_multipliers = {
            'nose': 0.5, 'left_eye': 0.5, 'right_eye': 0.5,
            'left_ear': 0.5, 'right_ear': 0.5,
            'left_shoulder': 0.7, 'right_shoulder': 0.7,
            'left_hip': 0.6, 'right_hip': 0.6,
            'left_elbow': 1.0, 'right_elbow': 1.0,
            'left_knee': 0.8, 'right_knee': 0.8,
            'left_wrist': 1.5, 'right_wrist': 1.5,  # Hands move most
            'left_ankle': 1.2, 'right_ankle': 1.2,
        }
        
        for name in KEYPOINT_NAMES:
            noise_mult = self.noise_multipliers.get(name, 1.0)
            self.filters[name] = KeypointKalmanFilter(
                process_noise=process_noise * noise_mult,
                measurement_noise=measurement_noise
            )
            self.confidences[name] = 0.0
    
    def update(self, keypoints: list) -> list:
        """Update filters with new keypoints and return smoothed results."""
        new_kp = {kp['name']: kp for kp in keypoints}
        smoothed = []
        
        for name in KEYPOINT_NAMES:
            kf = self.filters[name]
            kp = new_kp.get(name)
            
            # Always predict first (advance the state)
            predicted_pos = kf.predict()
            
            if kp and kp['confidence'] >= self.min_confidence:
                # Update with measurement
                smoothed_pos = kf.update(
                    [kp['x'], kp['y']], 
                    confidence=kp['confidence']
                )
                # EMA on confidence for smooth transitions
                self.confidences[name] = 0.6 * kp['confidence'] + 0.4 * self.confidences[name]
            else:
                # No measurement - use prediction
                smoothed_pos = predicted_pos
                # Decay confidence
                self.confidences[name] *= 0.85
            
            # Determine visibility
            visible = (
                kf.initialized and 
                kf.frames_since_measurement < self.persistence_frames and
                self.confidences[name] >= self.display_confidence
            )
            
            smoothed.append({
                'name': name,
                'x': int(smoothed_pos[0]),
                'y': int(smoothed_pos[1]),
                'confidence': self.confidences[name],
                'visible': visible,
                'frames_since_seen': kf.frames_since_measurement,
                'velocity': kf.get_velocity().tolist() if kf.initialized else [0, 0]
            })
        
        return smoothed
    
    def reset(self):
        """Reset all filters."""
        for name in KEYPOINT_NAMES:
            self.filters[name].reset()
            self.confidences[name] = 0.0
    
    def set_noise(self, process_noise=None, measurement_noise=None):
        """Adjust noise parameters for all filters."""
        for name, kf in self.filters.items():
            if process_noise is not None:
                noise_mult = self.noise_multipliers.get(name, 1.0)
                kf.Q = np.eye(4) * process_noise * noise_mult
                kf.Q[2, 2] *= 2
                kf.Q[3, 3] *= 2
            if measurement_noise is not None:
                kf.R = np.eye(2) * measurement_noise


class EMASmoother:
    """Simple EMA-based smoother (for comparison)."""
    
    def __init__(self, alpha=0.4, min_confidence=0.15, 
                 display_confidence=0.2, persistence_frames=5):
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.display_confidence = display_confidence
        self.persistence_frames = persistence_frames
        
        self.positions = {name: None for name in KEYPOINT_NAMES}
        self.confidences = {name: 0.0 for name in KEYPOINT_NAMES}
        self.frames_since_seen = {name: 999 for name in KEYPOINT_NAMES}
    
    def update(self, keypoints: list) -> list:
        new_kp = {kp['name']: kp for kp in keypoints}
        smoothed = []
        
        for name in KEYPOINT_NAMES:
            kp = new_kp.get(name)
            
            if kp and kp['confidence'] >= self.min_confidence:
                if self.positions[name] is None:
                    self.positions[name] = (kp['x'], kp['y'])
                else:
                    old_x, old_y = self.positions[name]
                    self.positions[name] = (
                        self.alpha * kp['x'] + (1 - self.alpha) * old_x,
                        self.alpha * kp['y'] + (1 - self.alpha) * old_y
                    )
                self.confidences[name] = 0.5 * kp['confidence'] + 0.5 * self.confidences[name]
                self.frames_since_seen[name] = 0
            else:
                self.frames_since_seen[name] += 1
                self.confidences[name] *= 0.85
            
            visible = (
                self.positions[name] is not None and
                self.frames_since_seen[name] < self.persistence_frames and
                self.confidences[name] >= self.display_confidence
            )
            
            if self.positions[name]:
                x, y = self.positions[name]
                smoothed.append({
                    'name': name,
                    'x': int(x),
                    'y': int(y),
                    'confidence': self.confidences[name],
                    'visible': visible,
                    'frames_since_seen': self.frames_since_seen[name]
                })
            else:
                smoothed.append({
                    'name': name, 'x': 0, 'y': 0,
                    'confidence': 0, 'visible': False,
                    'frames_since_seen': 999
                })
        
        return smoothed
    
    def reset(self):
        self.positions = {name: None for name in KEYPOINT_NAMES}
        self.confidences = {name: 0.0 for name in KEYPOINT_NAMES}
        self.frames_since_seen = {name: 999 for name in KEYPOINT_NAMES}


def draw_detections(frame, detections):
    """Draw bounding boxes for detected objects."""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = COLORS.get(det['class_name'], COLORS['default'])
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{det['class_name']} {det['confidence']:.0%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def draw_pose(frame, keypoints, show_confidence=False, show_velocity=False):
    """Draw pose skeleton and keypoints with color coding."""
    kp_dict = {kp['name']: kp for kp in keypoints}
    
    # Draw skeleton lines
    for start, end in SKELETON_CONNECTIONS:
        if start in kp_dict and end in kp_dict:
            kp1, kp2 = kp_dict[start], kp_dict[end]
            if kp1['visible'] and kp2['visible']:
                avg_conf = (kp1['confidence'] + kp2['confidence']) / 2
                intensity = int(155 + 100 * min(avg_conf, 1.0))
                color = (0, intensity, intensity)
                
                thickness = 3 if (kp1.get('frames_since_seen', 0) == 0 and 
                                  kp2.get('frames_since_seen', 0) == 0) else 2
                
                cv2.line(frame, (kp1['x'], kp1['y']), (kp2['x'], kp2['y']), color, thickness)
    
    # Draw keypoints
    for kp in keypoints:
        if kp['visible']:
            x, y = kp['x'], kp['y']
            name = kp['name']
            conf = kp['confidence']
            
            base_color = KEYPOINT_COLORS.get(name, (255, 255, 255))
            
            frames_old = kp.get('frames_since_seen', 0)
            if frames_old > 0:
                dim_factor = max(0.4, 1.0 - frames_old * 0.12)
                color = tuple(int(c * dim_factor) for c in base_color)
            else:
                color = base_color
            
            radius = int(4 + 4 * min(conf, 1.0))
            
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 1)
            
            # Draw velocity vector if available
            if show_velocity and 'velocity' in kp:
                vx, vy = kp['velocity']
                if abs(vx) > 0.5 or abs(vy) > 0.5:
                    end_x = int(x + vx * 5)
                    end_y = int(y + vy * 5)
                    cv2.arrowedLine(frame, (x, y), (end_x, end_y), (255, 255, 0), 2, tipLength=0.3)
            
            if show_confidence:
                cv2.putText(frame, f"{conf:.0%}", (x + 10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description='Visual Edge TPU Test with Kalman/EMA Smoothing')
    parser.add_argument('--camera', '-c', type=int, default=0)
    parser.add_argument('--width', '-W', type=int, default=640)
    parser.add_argument('--height', '-H', type=int, default=480)
    parser.add_argument('--stereo', '-s', action='store_true')
    parser.add_argument('--detection-only', '-d', action='store_true')
    parser.add_argument('--pose-only', '-p', action='store_true')
    parser.add_argument('--flip-h', action='store_true')
    parser.add_argument('--flip-v', action='store_true')
    parser.add_argument('--invert', '-i', action='store_true')
    
    # Smoothing options
    parser.add_argument('--filter', '-f', choices=['kalman', 'ema', 'none'], default='kalman',
                        help='Filter type: kalman (best), ema (simple), none (raw)')
    parser.add_argument('--process-noise', type=float, default=0.05,
                        help='Kalman process noise (lower=smoother, default: 0.05)')
    parser.add_argument('--measurement-noise', type=float, default=3.0,
                        help='Kalman measurement noise (higher=trust detection less, default: 3.0)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='EMA alpha (lower=smoother, default: 0.4)')
    parser.add_argument('--show-conf', action='store_true')
    parser.add_argument('--show-velocity', '-v', action='store_true',
                        help='Show velocity vectors (Kalman only)')
    
    args = parser.parse_args()
    
    print("=" * 65)
    print("🎥 Visual Edge TPU Test with Kalman/EMA Smoothing")
    print("=" * 65)
    print(f"Camera: {args.camera} @ {args.width}x{args.height}")
    print(f"Filter: {args.filter.upper()}")
    if args.filter == 'kalman':
        print(f"  Process noise: {args.process_noise}")
        print(f"  Measurement noise: {args.measurement_noise}")
    elif args.filter == 'ema':
        print(f"  Alpha: {args.alpha}")
    print()
    print("Controls:")
    print("  q      - Quit")
    print("  d/p/b  - Detection/Pose/Both mode")
    print("  1/2/3  - Kalman/EMA/None filter")
    print("  [ / ]  - Adjust smoothing (Kalman: process noise, EMA: alpha)")
    print("  , / .  - Adjust measurement noise (Kalman only)")
    print("  r      - Reset filter")
    print("  c      - Toggle confidence display")
    print("  v      - Toggle velocity vectors")
    print("=" * 65)
    
    # Connect to Edge TPU
    client = EdgeTPUClient()
    if not client.ping('detection'):
        print("❌ Edge TPU server not available!")
        return
    print("✓ Connected to Edge TPU server")
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width * (2 if args.stereo else 1))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"❌ Could not open camera {args.camera}")
        return
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera opened: {actual_w}x{actual_h}")
    
    # Initialize smoothers
    kalman_smoother = KalmanKeypointSmoother(
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise
    )
    ema_smoother = EMASmoother(alpha=args.alpha)
    
    # State
    run_detection = not args.pose_only
    run_pose = not args.detection_only
    current_filter = args.filter
    show_confidence = args.show_conf
    show_velocity = args.show_velocity
    process_noise = args.process_noise
    measurement_noise = args.measurement_noise
    ema_alpha = args.alpha
    
    fps_times = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            start_time = time.time()
            
            if args.invert or args.flip_h:
                frame = cv2.flip(frame, 1)
            if args.invert or args.flip_v:
                frame = cv2.flip(frame, 0)
            
            if args.stereo and actual_w > args.width:
                left_frame = frame[:, :actual_w//2]
                right_frame = frame[:, actual_w//2:]
                process_frame = left_frame
            else:
                process_frame = frame
                left_frame = frame
                right_frame = None
            
            det_time = 0
            pose_time = 0
            detections = []
            keypoints = []
            
            if run_detection:
                t = time.time()
                result = client.detect(process_frame, threshold=0.4)
                det_time = (time.time() - t) * 1000
                if result.get('success'):
                    detections = result.get('detections', [])
            
            if run_pose:
                t = time.time()
                result = client.estimate_pose(process_frame, threshold=0.1)
                pose_time = (time.time() - t) * 1000
                
                if result.get('success'):
                    raw_keypoints = result.get('keypoints', [])
                    
                    if current_filter == 'kalman':
                        keypoints = kalman_smoother.update(raw_keypoints)
                    elif current_filter == 'ema':
                        keypoints = ema_smoother.update(raw_keypoints)
                    else:
                        keypoints = [{**kp, 'visible': kp['confidence'] > 0.3} 
                                    for kp in raw_keypoints]
            
            # Draw
            if detections:
                draw_detections(process_frame, detections)
                if right_frame is not None:
                    draw_detections(right_frame, detections)
            
            if keypoints:
                draw_pose(process_frame, keypoints, show_confidence, 
                         show_velocity and current_filter == 'kalman')
                if right_frame is not None:
                    draw_pose(right_frame, keypoints, show_confidence,
                             show_velocity and current_filter == 'kalman')
            
            # FPS
            total_time = (time.time() - start_time) * 1000
            fps_times.append(total_time)
            if len(fps_times) > 30:
                fps_times.pop(0)
            avg_fps = 1000 / (sum(fps_times) / len(fps_times))
            
            display_frame = np.hstack([left_frame, right_frame]) if right_frame is not None else process_frame
            visible_kp = sum(1 for k in keypoints if k.get('visible', False))
            
            # Info overlay
            filter_info = {
                'kalman': f"KALMAN (p={process_noise:.3f} m={measurement_noise:.1f})",
                'ema': f"EMA (α={ema_alpha:.2f})",
                'none': "RAW (no filter)"
            }
            
            info_lines = [
                f"FPS: {avg_fps:.1f}",
                f"Filter: {filter_info[current_filter]}",
                f"Det: {det_time:.0f}ms | Pose: {pose_time:.0f}ms",
                f"Keypoints: {visible_kp}/17",
            ]
            
            y = 30
            for line in info_lines:
                cv2.putText(display_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(display_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                y += 20
            
            cv2.imshow("Edge TPU Vision Test", display_frame)
            
            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                run_detection, run_pose = True, False
            elif key == ord('p'):
                run_detection, run_pose = False, True
            elif key == ord('b'):
                run_detection, run_pose = True, True
            elif key == ord('1'):
                current_filter = 'kalman'
                print("Filter: KALMAN")
            elif key == ord('2'):
                current_filter = 'ema'
                print("Filter: EMA")
            elif key == ord('3'):
                current_filter = 'none'
                print("Filter: NONE (raw)")
            elif key == ord('['):
                if current_filter == 'kalman':
                    process_noise = max(0.001, process_noise * 0.7)
                    kalman_smoother.set_noise(process_noise=process_noise)
                    print(f"Process noise: {process_noise:.4f} (smoother)")
                else:
                    ema_alpha = max(0.1, ema_alpha - 0.05)
                    ema_smoother.alpha = ema_alpha
                    print(f"EMA alpha: {ema_alpha:.2f} (smoother)")
            elif key == ord(']'):
                if current_filter == 'kalman':
                    process_noise = min(1.0, process_noise * 1.4)
                    kalman_smoother.set_noise(process_noise=process_noise)
                    print(f"Process noise: {process_noise:.4f} (more responsive)")
                else:
                    ema_alpha = min(0.9, ema_alpha + 0.05)
                    ema_smoother.alpha = ema_alpha
                    print(f"EMA alpha: {ema_alpha:.2f} (more responsive)")
            elif key == ord(','):
                measurement_noise = max(0.5, measurement_noise * 0.7)
                kalman_smoother.set_noise(measurement_noise=measurement_noise)
                print(f"Measurement noise: {measurement_noise:.2f} (trust detections more)")
            elif key == ord('.'):
                measurement_noise = min(20.0, measurement_noise * 1.4)
                kalman_smoother.set_noise(measurement_noise=measurement_noise)
                print(f"Measurement noise: {measurement_noise:.2f} (trust detections less)")
            elif key == ord('r'):
                kalman_smoother.reset()
                ema_smoother.reset()
                print("Filters reset")
            elif key == ord('c'):
                show_confidence = not show_confidence
            elif key == ord('v'):
                show_velocity = not show_velocity
    
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")


if __name__ == '__main__':
    main()
