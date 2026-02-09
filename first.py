# import cv2
# import numpy as np
# from datetime import datetime
# import os
# from collections import deque

# class NervousnessDetector:
#     """
#     High-accuracy facial nervousness detection system using:
#     - Optical flow for precise micro-movements
#     - Statistical analysis for pattern detection
#     - Multi-scale feature tracking
#     - Temporal smoothing with outlier rejection
#     - Adaptive thresholding
#     """
    
#     def __init__(self, 
#                  nervousness_threshold=0.7,
#                  save_dir="nervous_captures",
#                  buffer_size=60):
#         """
#         Initialize the nervousness detector.
        
#         Args:
#             nervousness_threshold: Score above which image is saved (0-1)
#             save_dir: Directory to save captured images
#             buffer_size: Number of frames to analyze for patterns
#         """
#         self.threshold = nervousness_threshold
#         self.save_dir = save_dir
#         self.buffer_size = buffer_size
        
#         # Create save directory
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Load Haar Cascades
#         self.face_cascade = cv2.CascadeClassifier(
#             cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
#         )
#         self.eye_cascade = cv2.CascadeClassifier(
#             cv2.data.haarcascades + 'haarcascade_eye.xml'
#         )
        
#         # Advanced tracking variables
#         self.prev_gray = None
#         self.prev_face_roi = None
#         self.prev_face_pos = None
#         self.prev_face_size = None
#         self.prev_eyes = None
        
#         # Buffers for temporal analysis
#         self.movement_buffer = deque(maxlen=buffer_size)
#         self.optical_flow_buffer = deque(maxlen=buffer_size)
#         self.eye_aspect_ratio_buffer = deque(maxlen=buffer_size)
#         self.face_size_buffer = deque(maxlen=buffer_size)
#         self.head_pose_buffer = deque(maxlen=buffer_size)
#         self.nervousness_scores = deque(maxlen=buffer_size)
#         self.eye_blink_events = deque(maxlen=buffer_size)
        
#         # Calibration variables (adaptive baseline)
#         self.baseline_movement = None
#         self.baseline_flow = None
#         self.calibration_frames = 0
#         self.is_calibrated = False
        
#         # Frame counter
#         self.frame_count = 0
        
#         # Parameters for optical flow
#         self.lk_params = dict(
#             winSize=(15, 15),
#             maxLevel=2,
#             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
#         )
        
#     def detect_nervousness(self, frame):
#         """
#         Analyze frame for nervousness indicators with high accuracy.
        
#         Returns:
#             nervousness_score: Float between 0-1
#             annotated_frame: Frame with detection overlays
#             indicators: Dictionary of individual indicator scores
#             confidence: Confidence level of detection (0-1)
#         """
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Apply histogram equalization for better face detection
#         gray = cv2.equalizeHist(gray)
        
#         faces = self.face_cascade.detectMultiScale(
#             gray, 
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(80, 80),
#             flags=cv2.CASCADE_SCALE_IMAGE
#         )
        
#         nervousness_score = 0.0
#         confidence = 0.0
#         annotated = frame.copy()
#         indicators = {}
        
#         if len(faces) > 0:
#             # Get primary face (largest and most centered)
#             x, y, w, h = self._get_best_face(faces, frame.shape)
            
#             # Extract face ROI
#             face_roi = gray[y:y+h, x:x+w]
#             face_roi_color = frame[y:y+h, x:x+w]
            
#             # Draw face rectangle
#             cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
#             # === INDICATOR 1: Optical Flow Analysis ===
#             flow_score = self._calculate_optical_flow(gray, x, y, w, h)
#             indicators['optical_flow'] = flow_score
            
#             # === INDICATOR 2: Precise Head Movement Tracking ===
#             movement_score = self._calculate_precise_movement(x, y, w, h)
#             indicators['head_movement'] = movement_score
            
#             # === INDICATOR 3: Advanced Eye Blink Detection ===
#             eyes = self.eye_cascade.detectMultiScale(
#                 face_roi,
#                 scaleFactor=1.1,
#                 minNeighbors=10,
#                 minSize=(20, 20)
#             )
#             blink_score = self._calculate_advanced_blinks(eyes, face_roi)
#             indicators['blink_rate'] = blink_score
            
#             # Draw eyes
#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(annotated, (x+ex, y+ey), 
#                             (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            
#             # === INDICATOR 4: Face Size Stability ===
#             size_score = self._calculate_size_stability(w, h)
#             indicators['position_stability'] = size_score
            
#             # === INDICATOR 5: Head Pose Variation ===
#             pose_score = self._calculate_head_pose_variation(x, y, w, h, frame.shape)
#             indicators['head_pose'] = pose_score
            
#             # === INDICATOR 6: Micro-Expression Detection ===
#             micro_score = self._calculate_micro_expressions(face_roi)
#             indicators['micro_expressions'] = micro_score
            
#             # === INDICATOR 7: Movement Frequency Analysis ===
#             frequency_score = self._calculate_movement_frequency()
#             indicators['movement_frequency'] = frequency_score
            
#             # Update calibration baseline
#             if not self.is_calibrated:
#                 self._update_calibration(indicators)
            
#             # Combine scores with adaptive weights
#             weights = self._get_adaptive_weights(indicators)
            
#             nervousness_score = sum(
#                 indicators[key] * weights[key] 
#                 for key in weights
#             )
            
#             # Calculate confidence
#             confidence = self._calculate_confidence(eyes, w, h)
            
#             # Apply temporal smoothing
#             self.nervousness_scores.append(nervousness_score)
#             smoothed_score = self._smooth_with_outlier_rejection()
            
#             # Adjust score based on calibration
#             if self.is_calibrated:
#                 smoothed_score = self._apply_baseline_adjustment(smoothed_score)
            
#             # Draw overlay
#             self._draw_advanced_overlay(annotated, smoothed_score, indicators, confidence)
            
#             self.prev_gray = gray
#             self.prev_face_roi = face_roi
#             self.prev_face_pos = (x, y, w, h)
#             self.frame_count += 1
            
#             return smoothed_score, annotated, indicators, confidence
        
#         self.prev_gray = gray
#         return 0.0, annotated, {}, 0.0
    
#     def _get_best_face(self, faces, frame_shape):
#         """Select the most relevant face."""
#         h, w = frame_shape[:2]
#         center_x, center_y = w // 2, h // 2
        
#         best_face = None
#         best_score = -1
        
#         for (x, y, fw, fh) in faces:
#             size_score = fw * fh
#             face_center_x = x + fw // 2
#             face_center_y = y + fh // 2
#             distance = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
#             center_score = 1.0 / (1.0 + distance / 100)
#             score = size_score * center_score
            
#             if score > best_score:
#                 best_score = score
#                 best_face = (x, y, fw, fh)
        
#         return best_face
    
#     def _calculate_optical_flow(self, gray, x, y, w, h):
#         """Calculate optical flow for micro-movement detection."""
#         if self.prev_gray is None or self.prev_face_roi is None:
#             return 0.0
        
#         try:
#             face_roi = gray[y:y+h, x:x+w]
            
#             corners = cv2.goodFeaturesToTrack(
#                 self.prev_face_roi,
#                 maxCorners=50,
#                 qualityLevel=0.01,
#                 minDistance=10
#             )
            
#             if corners is None or len(corners) < 10:
#                 return 0.0
            
#             next_points, status, _ = cv2.calcOpticalFlowPyrLK(
#                 self.prev_face_roi, face_roi, corners, None, **self.lk_params
#             )
            
#             if next_points is None:
#                 return 0.0
            
#             good_old = corners[status == 1]
#             good_new = next_points[status == 1]
            
#             if len(good_new) < 5:
#                 return 0.0
            
#             movements = np.linalg.norm(good_new - good_old, axis=1)
#             avg_movement = np.mean(movements)
            
#             self.optical_flow_buffer.append(avg_movement)
            
#             if len(self.optical_flow_buffer) >= 20:
#                 recent_flow = list(self.optical_flow_buffer)[-20:]
#                 flow_mean = np.mean(recent_flow)
#                 flow_std = np.std(recent_flow)
#                 score = min((flow_mean / 3.0) * (1 + flow_std / 2.0), 1.0)
#                 return score
            
#             return 0.0
            
#         except:
#             return 0.0
    
#     def _calculate_precise_movement(self, x, y, w, h):
#         """Calculate precise head movement."""
#         current_pos = (x + w//2, y + h//2)
        
#         if self.prev_face_pos is None:
#             return 0.0
        
#         px, py, pw, ph = self.prev_face_pos
#         prev_pos = (px + pw//2, py + ph//2)
        
#         dx = current_pos[0] - prev_pos[0]
#         dy = current_pos[1] - prev_pos[1]
#         displacement = np.sqrt(dx**2 + dy**2)
        
#         self.movement_buffer.append(displacement)
        
#         if len(self.movement_buffer) >= 30:
#             recent = list(self.movement_buffer)[-30:]
#             mean_movement = np.mean(recent)
#             std_movement = np.std(recent)
            
#             direction_changes = 0
#             for i in range(1, len(recent)-1):
#                 if recent[i] > recent[i-1] and recent[i] > recent[i+1]:
#                     direction_changes += 1
            
#             movement_score = min(mean_movement / 8.0, 1.0)
#             variance_score = min(std_movement / 5.0, 1.0)
#             fidget_score = min(direction_changes / 10.0, 1.0)
            
#             return (movement_score * 0.4 + variance_score * 0.3 + fidget_score * 0.3)
        
#         return 0.0
    
#     def _calculate_advanced_blinks(self, eyes, face_roi):
#         """Advanced eye blink detection."""
#         current_eye_count = len(eyes)
#         self.eye_blink_events.append(current_eye_count)
        
#         if len(self.eye_blink_events) < 30:
#             return 0.0
        
#         recent = list(self.eye_blink_events)[-30:]
        
#         blinks = 0
#         for i in range(1, len(recent)):
#             if recent[i-1] >= 2 and recent[i] < 2:
#                 blinks += 1
        
#         blink_rate = blinks / 1.0
#         prolonged_closures = sum(1 for count in recent if count == 0)
        
#         blink_score = min(blink_rate / 0.5, 1.0) * 0.7
#         closure_score = min(prolonged_closures / 10.0, 1.0) * 0.3
        
#         return blink_score + closure_score
    
#     def _calculate_size_stability(self, w, h):
#         """Detect face size variations."""
#         current_size = w * h
        
#         if self.prev_face_size is None:
#             self.prev_face_size = current_size
#             return 0.0
        
#         size_change = abs(current_size - self.prev_face_size) / self.prev_face_size
#         self.face_size_buffer.append(size_change)
#         self.prev_face_size = current_size
        
#         if len(self.face_size_buffer) >= 30:
#             recent = list(self.face_size_buffer)[-30:]
#             mean_change = np.mean(recent)
#             significant_changes = sum(1 for c in recent if c > 0.02)
            
#             stability_score = min(mean_change * 30, 1.0) * 0.6
#             frequency_score = min(significant_changes / 15.0, 1.0) * 0.4
            
#             return stability_score + frequency_score
        
#         return 0.0
    
#     def _calculate_head_pose_variation(self, x, y, w, h, frame_shape):
#         """Analyze head pose stability."""
#         fh, fw = frame_shape[:2]
#         rel_x = (x + w/2) / fw
#         rel_y = (y + h/2) / fh
        
#         self.head_pose_buffer.append((rel_x, rel_y))
        
#         if len(self.head_pose_buffer) >= 30:
#             recent = list(self.head_pose_buffer)[-30:]
#             x_positions = [pos[0] for pos in recent]
#             y_positions = [pos[1] for pos in recent]
            
#             x_std = np.std(x_positions)
#             y_std = np.std(y_positions)
            
#             pose_score = min((x_std + y_std) * 30, 1.0)
#             return pose_score
        
#         return 0.0
    
#     def _calculate_micro_expressions(self, face_roi):
#         """Detect micro-expressions using texture analysis."""
#         if face_roi is None or face_roi.size == 0:
#             return 0.0
        
#         try:
#             laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
#             variance = laplacian.var()
#             return min(variance / 1000.0, 1.0)
#         except:
#             return 0.0
    
#     def _calculate_movement_frequency(self):
#         """Analyze frequency of movements using FFT."""
#         if len(self.movement_buffer) < 30:
#             return 0.0
        
#         recent = list(self.movement_buffer)[-30:]
#         fft = np.fft.fft(recent)
#         power = np.abs(fft[:len(fft)//2])
        
#         high_freq_power = np.sum(power[5:15])
#         total_power = np.sum(power) + 0.001
        
#         frequency_score = min(high_freq_power / total_power * 3, 1.0)
#         return frequency_score
    
#     def _update_calibration(self, indicators):
#         """Update baseline calibration."""
#         self.calibration_frames += 1
        
#         if self.calibration_frames == 60:
#             if len(self.optical_flow_buffer) > 0:
#                 self.baseline_flow = np.mean(list(self.optical_flow_buffer))
#             if len(self.movement_buffer) > 0:
#                 self.baseline_movement = np.mean(list(self.movement_buffer))
            
#             self.is_calibrated = True
#             print("✓ Calibration complete - baseline established")
    
#     def _get_adaptive_weights(self, indicators):
#         """Get adaptive weights based on detection confidence."""
#         weights = {
#             'optical_flow': 0.25,
#             'head_movement': 0.20,
#             'blink_rate': 0.15,
#             'position_stability': 0.15,
#             'head_pose': 0.10,
#             'micro_expressions': 0.08,
#             'movement_frequency': 0.07
#         }
        
#         if indicators.get('optical_flow', 0) == 0:
#             weights['head_movement'] += 0.15
#             weights['optical_flow'] = 0.10
        
#         return weights
    
#     def _calculate_confidence(self, eyes, w, h):
#         """Calculate detection confidence."""
#         confidence = 0.5
        
#         if 100 < w < 400:
#             confidence += 0.2
        
#         if len(eyes) >= 2:
#             confidence += 0.2
        
#         if len(self.movement_buffer) >= 30:
#             confidence += 0.1
        
#         return min(confidence, 1.0)
    
#     def _smooth_with_outlier_rejection(self):
#         """Apply temporal smoothing with outlier rejection."""
#         if len(self.nervousness_scores) < 10:
#             return np.mean(self.nervousness_scores) if len(self.nervousness_scores) > 0 else 0.0
        
#         recent = list(self.nervousness_scores)[-20:]
        
#         q1, q3 = np.percentile(recent, [25, 75])
#         iqr = q3 - q1
#         lower_bound = q1 - 1.5 * iqr
#         upper_bound = q3 + 1.5 * iqr
        
#         filtered = [x for x in recent if lower_bound <= x <= upper_bound]
        
#         if len(filtered) == 0:
#             return np.mean(recent)
        
#         weights = np.exp(np.linspace(-1, 0, len(filtered)))
#         weighted_avg = np.average(filtered, weights=weights)
        
#         return weighted_avg
    
#     def _apply_baseline_adjustment(self, score):
#         """Adjust score based on calibrated baseline."""
#         if self.baseline_movement and len(self.movement_buffer) > 0:
#             current_movement = np.mean(list(self.movement_buffer)[-10:])
#             if current_movement < self.baseline_movement * 1.5:
#                 score *= 0.8
        
#         return score
    
#     def _draw_advanced_overlay(self, frame, score, indicators, confidence):
#         """Draw comprehensive information overlay."""
#         h, w = frame.shape[:2]
        
#         # Semi-transparent overlay panel
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (5, 5), (420, 280), (0, 0, 0), -1)
#         cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
#         # Status header
#         status = "NERVOUS" if score > self.threshold else "CALM"
#         color = (0, 0, 255) if score > self.threshold else (0, 255, 0)
        
#         cv2.putText(frame, f"Status: {status}", (15, 35),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#         cv2.putText(frame, f"Score: {score:.3f}", (15, 70),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#         cv2.putText(frame, f"Confidence: {confidence:.2f}", (15, 100),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
#         # Calibration status
#         if not self.is_calibrated:
#             cv2.putText(frame, f"Calibrating... {self.calibration_frames}/60", 
#                        (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
#         # Individual indicators
#         y_pos = 160
#         for name, value in indicators.items():
#             label = name.replace('_', ' ').title()[:20]
            
#             if value < 0.4:
#                 bar_color = (0, 255, 0)
#             elif value < 0.7:
#                 bar_color = (0, 165, 255)
#             else:
#                 bar_color = (0, 0, 255)
            
#             cv2.putText(frame, f"{label}:", (15, y_pos),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
#             bar_width = int(value * 180)
#             cv2.rectangle(frame, (200, y_pos-12), (200 + bar_width, y_pos-2),
#                          bar_color, -1)
#             cv2.rectangle(frame, (200, y_pos-12), (380, y_pos-2),
#                          (100, 100, 100), 1)
            
#             cv2.putText(frame, f"{value:.2f}", (385, y_pos),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
#             y_pos += 20
    
#     def save_capture(self, frame, score, indicators=None, confidence=0.0):
#         """Save frame with comprehensive metadata."""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         state = "nervous" if score > self.threshold else "calm"
#         filename = f"{state}_{timestamp}_s{score:.2f}_c{confidence:.2f}.jpg"
#         filepath = os.path.join(self.save_dir, filename)
        
#         cv2.imwrite(filepath, frame)
        
#         if indicators:
#             meta_file = filepath.replace('.jpg', '_meta.txt')
#             with open(meta_file, 'w') as f:
#                 f.write(f"Timestamp: {timestamp}\n")
#                 f.write(f"State: {state.upper()}\n")
#                 f.write(f"Overall Score: {score:.3f}\n")
#                 f.write(f"Confidence: {confidence:.2f}\n")
#                 f.write(f"Calibrated: {self.is_calibrated}\n")
#                 f.write("\nDetailed Indicators:\n")
#                 for key, value in sorted(indicators.items()):
#                     f.write(f"  {key}: {value:.3f}\n")
                
#                 f.write("\nInterpretation:\n")
#                 if score < 0.3:
#                     f.write("  Very calm and relaxed state\n")
#                 elif score < 0.5:
#                     f.write("  Normal/neutral state\n")
#                 elif score < 0.7:
#                     f.write("  Mild nervousness detected\n")
#                 else:
#                     f.write("  Significant nervousness indicators\n")
        
#         print(f"✓ Saved: {filename} | Score: {score:.3f} | Confidence: {confidence:.2f}")
#         return filepath


# def main():
#     """
#     Main function - ONLY saves images when NERVOUS (not when calm).
#     """
#     # Initialize detector
#     detector = NervousnessDetector(
#         nervousness_threshold=0.5,
#         save_dir="nervous_captures",
#         buffer_size=60
#     )
    
#     # Open webcam
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FPS, 30)
    
#     if not cap.isOpened():
#         print("Error: Could not open webcam")
#         return
    
#     print("=" * 60)
#     print("HIGH-ACCURACY NERVOUSNESS DETECTION SYSTEM")
#     print("=" * 60)
#     print("Advanced Analysis Methods:")
#     print("  • Optical flow tracking (micro-movements)")
#     print("  • Statistical pattern analysis")
#     print("  • Eye aspect ratio monitoring")
#     print("  • Head pose variation")
#     print("  • Movement frequency analysis")
#     print("  • Adaptive baseline calibration")
#     print("\n⚠️  SAVE MODE: NERVOUS ONLY (Calm states ignored)")
#     print("\nPress 'q' to quit, 'r' to recalibrate")
#     print("=" * 60)
    
#     last_save_time = 0
#     save_cooldown = 3
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         score, annotated, indicators, confidence = detector.detect_nervousness(frame)
        
#         current_time = datetime.now().timestamp()
        
#         # ONLY save when NERVOUS (score > threshold) and confident
#         if score > detector.threshold and confidence > 0.5:
#             if current_time - last_save_time > save_cooldown:
#                 detector.save_capture(frame, score, indicators, confidence)
#                 last_save_time = current_time
        
#         cv2.imshow('High-Accuracy Nervousness Detector', annotated)
        
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('r'):
#             detector.is_calibrated = False
#             detector.calibration_frames = 0
#             print("\n♻ Recalibrating...")
    
#     cap.release()
#     cv2.destroyAllWindows()
#     print("\n✓ Detection session ended")


# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
from datetime import datetime
import os
from collections import deque
import base64
import time

class NervousnessDetector:
    """
    High-accuracy facial nervousness detection system using:
    - Optical flow for precise micro-movements
    - Statistical analysis for pattern detection
    - Multi-scale feature tracking
    - Temporal smoothing with outlier rejection
    - Adaptive thresholding
    """
    
    def __init__(self, 
                 nervousness_threshold=0.5,
                 save_dir="nervous_captures",
                 buffer_size=8):
        """
        Initialize the nervousness detector.
        
        Args:
            nervousness_threshold: Score above which image is saved (0-1)
            save_dir: Directory to save captured images
            buffer_size: Number of frames to analyze for patterns
        """
        self.threshold = nervousness_threshold
        self.save_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), save_dir)
)
        self.buffer_size = buffer_size
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load Haar Cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Try to load smile cascade for mouth detection
        try:
            self.mouth_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_smile.xml'
            )
        except:
            self.mouth_cascade = None
        
        # Advanced tracking variables
        self.prev_gray = None
        self.prev_face_roi = None
        self.prev_face_pos = None
        self.prev_face_size = None
        self.prev_eyes = None
        
        # Buffers for temporal analysis
        self.movement_buffer = deque(maxlen=buffer_size)
        self.optical_flow_buffer = deque(maxlen=buffer_size)
        self.eye_aspect_ratio_buffer = deque(maxlen=buffer_size)
        self.face_size_buffer = deque(maxlen=buffer_size)
        self.head_pose_buffer = deque(maxlen=buffer_size)
        self.nervousness_scores = deque(maxlen=buffer_size)
        self.eye_blink_events = deque(maxlen=buffer_size)
        self.lip_movement_buffer = deque(maxlen=buffer_size)
        self.mouth_aspect_ratio_buffer = deque(maxlen=buffer_size)
        
        # Calibration variables (adaptive baseline)
        self.baseline_movement = None
        self.baseline_flow = None
        self.calibration_frames = 0
        self.is_calibrated = False
        
        # Frame counter
        self.frame_count = 0
        
        # Track saved images to avoid duplicates
        self.saved_images = []  # List of (score, filepath, timestamp)
        self.score_similarity_threshold = 0.01  # Only consider nearly identical scores (0.01 difference)
        
        # Parameters for optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        self.recent_peak_score = 0.0
        self.last_capture_time = 0
        self.capture_cooldown = 5
        
    def detect_nervousness(self, frame):
        """
        Analyze frame for nervousness indicators with high accuracy.
        
        Returns:
            nervousness_score: Float between 0-1
            annotated_frame: Frame with detection overlays
            indicators: Dictionary of individual indicator scores
            confidence: Confidence level of detection (0-1)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better face detection
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        nervousness_score = 0.0
        confidence = 0.0
        annotated = frame.copy()
        indicators = {}
        
        if len(faces) > 0:
            # Get primary face (largest and most centered)
            x, y, w, h = self._get_best_face(faces, frame.shape)
            
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]
            
            # Draw face rectangle
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # === INDICATOR 1: Optical Flow Analysis ===
            flow_score = self._calculate_optical_flow(gray, x, y, w, h)
            indicators['optical_flow'] = flow_score
            
            # === INDICATOR 2: Precise Head Movement Tracking ===
            movement_score = self._calculate_precise_movement(x, y, w, h)
            indicators['head_movement'] = movement_score
            
            # === INDICATOR 3: Advanced Eye Blink Detection ===
            eyes = self.eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(20, 20)
            )
            blink_score = self._calculate_advanced_blinks(eyes, face_roi)
            indicators['blink_rate'] = blink_score
            
            # Draw eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(annotated, (x+ex, y+ey), 
                            (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            
            # === INDICATOR 4: Face Size Stability ===
            size_score = self._calculate_size_stability(w, h)
            indicators['position_stability'] = size_score
            
            # === INDICATOR 5: Head Pose Variation ===
            pose_score = self._calculate_head_pose_variation(x, y, w, h, frame.shape)
            indicators['head_pose'] = pose_score
            
            # === INDICATOR 6: Micro-Expression Detection ===
            micro_score = self._calculate_micro_expressions(face_roi)
            indicators['micro_expressions'] = micro_score
            
            # === INDICATOR 7: Movement Frequency Analysis ===
            frequency_score = self._calculate_movement_frequency()
            indicators['movement_frequency'] = frequency_score
            
            # === INDICATOR 8: Lip/Mouth Movement Analysis ===
            lip_score = self._calculate_lip_movement(face_roi, x, y, w, h, annotated)
            indicators['lip_movement'] = lip_score
            
            # Update calibration baseline
            if not self.is_calibrated:
                self._update_calibration(indicators)
            
            # Combine scores with adaptive weights
            weights = self._get_adaptive_weights(indicators)
            
            nervousness_score = sum(
                indicators[key] * weights[key] 
                for key in weights
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(eyes, w, h)
            
            # Apply temporal smoothing
            self.nervousness_scores.append(nervousness_score)
            smoothed_score = self._smooth_with_outlier_rejection()
            
            # Adjust score based on calibration
            if self.is_calibrated:
                smoothed_score = self._apply_baseline_adjustment(smoothed_score)
            
            # Draw overlay
            self._draw_advanced_overlay(annotated, smoothed_score, indicators, confidence)
            
            self.prev_gray = gray
            self.prev_face_roi = face_roi
            self.prev_face_pos = (x, y, w, h)
            self.frame_count += 1
            
            return smoothed_score, annotated, indicators, confidence
        
        self.prev_gray = gray
        return 0.0, annotated, {}, 0.0
    
    
    def analyze_frame_for_api(self, frame):
        try:
            score, annotated, indicators, confidence = self.detect_nervousness(frame)

            image_base64 = None
            nervous = score > self.threshold

            # Only send image when it actually matters
            if nervous and confidence > 0.1:
                _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                image_base64 = base64.b64encode(buffer).decode("utf-8")

            return {
                "nervous": bool(nervous),
                "score": float(score),
                "confidence": float(confidence),
                "imageBase64": image_base64
            }

        except Exception as e:
            print("❌ analyze_frame_for_api failed:", e)
            return {
                "nervous": False,
                "score": 0.0,
                "confidence": 0.0,
                "imageBase64": None
            }


       

   
    def _get_best_face(self, faces, frame_shape):
        """Select the most relevant face."""
        h, w = frame_shape[:2]
        center_x, center_y = w // 2, h // 2
        
        best_face = None
        best_score = -1
        
        for (x, y, fw, fh) in faces:
            size_score = fw * fh
            face_center_x = x + fw // 2
            face_center_y = y + fh // 2
            distance = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
            center_score = 1.0 / (1.0 + distance / 100)
            score = size_score * center_score
            
            if score > best_score:
                best_score = score
                best_face = (x, y, fw, fh)
        
        return best_face
    
    def _calculate_optical_flow(self, gray, x, y, w, h):
        """Calculate optical flow for micro-movement detection."""
        if self.prev_gray is None or self.prev_face_roi is None:
            return 0.0
        
        try:
            face_roi = gray[y:y+h, x:x+w]
            
            corners = cv2.goodFeaturesToTrack(
                self.prev_face_roi,
                maxCorners=50,
                qualityLevel=0.01,
                minDistance=10
            )
            
            if corners is None or len(corners) < 10:
                return 0.0
            
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_face_roi, face_roi, corners, None, **self.lk_params
            )
            
            if next_points is None:
                return 0.0
            
            good_old = corners[status == 1]
            good_new = next_points[status == 1]
            
            if len(good_new) < 5:
                return 0.0
            
            movements = np.linalg.norm(good_new - good_old, axis=1)
            avg_movement = np.mean(movements)
            
            self.optical_flow_buffer.append(avg_movement)
            
            if len(self.optical_flow_buffer) >= 5:
                recent_flow = list(self.optical_flow_buffer)[-5:]
                flow_mean = np.mean(recent_flow)
                flow_std = np.std(recent_flow)
                score = min((flow_mean / 3.0) * (1 + flow_std / 2.0), 1.0)
                return score
            
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_precise_movement(self, x, y, w, h):
        """Calculate precise head movement."""
        current_pos = (x + w//2, y + h//2)
        
        if self.prev_face_pos is None:
            return 0.0
        
        px, py, pw, ph = self.prev_face_pos
        prev_pos = (px + pw//2, py + ph//2)
        
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        displacement = np.sqrt(dx**2 + dy**2)
        
        self.movement_buffer.append(displacement)
        
        if len(self.movement_buffer) >= 5:
            recent = list(self.movement_buffer)[-5:]
            mean_movement = np.mean(recent)
            std_movement = np.std(recent)
            
            direction_changes = 0
            for i in range(1, len(recent)-1):
                if recent[i] > recent[i-1] and recent[i] > recent[i+1]:
                    direction_changes += 1
            
            movement_score = min(mean_movement / 8.0, 1.0)
            variance_score = min(std_movement / 5.0, 1.0)
            fidget_score = min(direction_changes / 10.0, 1.0)
            
            return (movement_score * 0.4 + variance_score * 0.3 + fidget_score * 0.3)
        
        return 0.0
    
    def _calculate_advanced_blinks(self, eyes, face_roi):
        """Advanced eye blink detection."""
        current_eye_count = len(eyes)
        self.eye_blink_events.append(current_eye_count)
        
        if len(self.eye_blink_events) < 5:
            return 0.0
        
        recent = list(self.eye_blink_events)[-5:]
        
        blinks = 0
        for i in range(1, len(recent)):
            if recent[i-1] >= 2 and recent[i] < 2:
                blinks += 1
        
        blink_rate = blinks / 1.0
        prolonged_closures = sum(1 for count in recent if count == 0)
        
        blink_score = min(blink_rate / 0.5, 1.0) * 0.7
        closure_score = min(prolonged_closures / 10.0, 1.0) * 0.3
        
        return blink_score + closure_score
    
    def _calculate_size_stability(self, w, h):
        """Detect face size variations."""
        current_size = w * h
        
        if self.prev_face_size is None:
            self.prev_face_size = current_size
            return 0.0
        
        size_change = abs(current_size - self.prev_face_size) / self.prev_face_size
        self.face_size_buffer.append(size_change)
        self.prev_face_size = current_size
        
        if len(self.face_size_buffer) >= 5:
            recent = list(self.face_size_buffer)[-5:]
            mean_change = np.mean(recent)
            significant_changes = sum(1 for c in recent if c > 0.02)
            
            stability_score = min(mean_change * 30, 1.0) * 0.6
            frequency_score = min(significant_changes / 15.0, 1.0) * 0.4
            
            return stability_score + frequency_score
        
        return 0.0
    
    def _calculate_head_pose_variation(self, x, y, w, h, frame_shape):
        """Analyze head pose stability."""
        fh, fw = frame_shape[:2]
        rel_x = (x + w/2) / fw
        rel_y = (y + h/2) / fh
        
        self.head_pose_buffer.append((rel_x, rel_y))
        
        if len(self.head_pose_buffer) >= 5:
            recent = list(self.head_pose_buffer)[-5:]
            x_positions = [pos[0] for pos in recent]
            y_positions = [pos[1] for pos in recent]
            
            x_std = np.std(x_positions)
            y_std = np.std(y_positions)
            
            pose_score = min((x_std + y_std) * 30, 1.0)
            return pose_score
        
        return 0.0
    
    def _calculate_micro_expressions(self, face_roi):
        """Detect micro-expressions using texture analysis."""
        if face_roi is None or face_roi.size == 0:
            return 0.0
        
        try:
            laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
            variance = laplacian.var()
            return min(variance / 1000.0, 1.0)
        except:
            return 0.0
    
    def _calculate_movement_frequency(self):
        """Analyze frequency of movements using FFT."""
        if len(self.movement_buffer) < 5:
            return 0.0
        
        recent = list(self.movement_buffer)[-5:]
        fft = np.fft.fft(recent)
        power = np.abs(fft[:len(fft)//2])
        
        high_freq_power = np.sum(power[5:15])
        total_power = np.sum(power) + 0.001
        
        frequency_score = min(high_freq_power / total_power * 3, 1.0)
        return frequency_score
    
    def _calculate_lip_movement(self, face_roi, face_x, face_y, face_w, face_h, annotated):
        """
        Detect lip/mouth movements indicating nervousness:
        - Lip biting, pursing
        - Excessive mouth movements
        - Mouth fidgeting
        - Talking frequency
        """
        if face_roi is None or face_roi.size == 0:
            return 0.0
        
        try:
            # Define mouth region (lower half of face)
            mouth_y_start = int(face_h * 0.6)
            mouth_y_end = int(face_h * 0.95)
            mouth_x_start = int(face_w * 0.25)
            mouth_x_end = int(face_w * 0.75)
            
            mouth_roi = face_roi[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
            
            if mouth_roi.size == 0:
                return 0.0
            
            # Method 1: Detect mouth using smile cascade
            mouth_movements = 0
            if self.mouth_cascade is not None:
                mouths = self.mouth_cascade.detectMultiScale(
                    mouth_roi,
                    scaleFactor=1.8,
                    minNeighbors=15,
                    minSize=(25, 15)
                )
                
                # Draw detected mouth
                for (mx, my, mw, mh) in mouths:
                    abs_mx = face_x + mouth_x_start + mx
                    abs_my = face_y + mouth_y_start + my
                    cv2.rectangle(annotated, (abs_mx, abs_my), 
                                (abs_mx + mw, abs_my + mh), (255, 0, 255), 2)
                
                mouth_movements = len(mouths)
            
            # Method 2: Analyze mouth region texture/variance (movement detection)
            # Calculate horizontal edges (lip movements create strong horizontal edges)
            sobelx = cv2.Sobel(mouth_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(mouth_roi, cv2.CV_64F, 0, 1, ksize=3)
            
            # Horizontal edges indicate lip line
            horizontal_edges = np.abs(sobely).mean()
            
            # Method 3: Calculate "Mouth Aspect Ratio" (MAR) - inspired by EAR
            # Estimate mouth openness using vertical variance in mouth region
            vertical_profile = np.mean(mouth_roi, axis=1)
            mouth_variance = np.var(vertical_profile)
            
            # Normalize mouth variance
            mouth_activity = min(mouth_variance / 500.0, 1.0)
            
            self.mouth_aspect_ratio_buffer.append(mouth_activity)
            
            if len(self.mouth_aspect_ratio_buffer) >= 5:
                recent_mouth = list(self.mouth_aspect_ratio_buffer)[-5:]
                
                # Calculate mouth movement statistics
                mouth_mean = np.mean(recent_mouth)
                mouth_std = np.std(recent_mouth)
                
                # Count significant changes (talking, biting, pursing)
                mouth_changes = 0
                for i in range(1, len(recent_mouth)):
                    if abs(recent_mouth[i] - recent_mouth[i-1]) > 0.1:
                        mouth_changes += 1
                
                # Nervous indicators:
                # 1. High average activity (excessive movement)
                # 2. High variance (inconsistent movements - fidgeting)
                # 3. Frequent changes (nervous talking, lip biting)
                
                activity_score = min(mouth_mean * 2.0, 1.0)
                variance_score = min(mouth_std * 3.0, 1.0)
                change_score = min(mouth_changes / 10.0, 1.0)
                
                # Combine scores
                lip_score = (activity_score * 0.4 + variance_score * 0.3 + change_score * 0.3)
                
                # Store for temporal analysis
                self.lip_movement_buffer.append(lip_score)
                
                return lip_score
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    def _update_calibration(self, indicators):
        """Update baseline calibration."""
        self.calibration_frames += 1
        
        if self.calibration_frames == 30:
            if len(self.optical_flow_buffer) > 0:
                self.baseline_flow = np.mean(list(self.optical_flow_buffer))
            if len(self.movement_buffer) > 0:
                self.baseline_movement = np.mean(list(self.movement_buffer))
            
            self.is_calibrated = True
            print("✓ Calibration complete - baseline established")
    
    def _get_adaptive_weights(self, indicators):
        """Get adaptive weights based on detection confidence."""
        weights = {
            'optical_flow': 0.28,
            'head_movement': 0.22,
            'blink_rate': 0.10,
            'lip_movement': 0.08,
            'position_stability': 0.14,
            'head_pose': 0.10,
            'micro_expressions': 0.05,
            'movement_frequency': 0.03
        }

        
        # If optical flow is unreliable, redistribute weight
        if indicators.get('optical_flow', 0) == 0:
            weights['head_movement'] += 0.12
            weights['optical_flow'] = 0.10
        
        # If lip detection fails, redistribute weight
        if indicators.get('lip_movement', 0) == 0:
            weights['blink_rate'] += 0.06
            weights['head_movement'] += 0.06
            weights['lip_movement'] = 0.0
        
        return weights
    
    def _calculate_confidence(self, eyes, w, h):
        """Calculate detection confidence."""
        confidence = 0.5
        
        if 100 < w < 400:
            confidence += 0.2
        
        if len(eyes) >= 2:
            confidence += 0.2
        
        if len(self.movement_buffer) >= 30:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _smooth_with_outlier_rejection(self):
        """Apply temporal smoothing with outlier rejection."""
        if len(self.nervousness_scores) < 10:
            return np.mean(self.nervousness_scores) if len(self.nervousness_scores) > 0 else 0.0
        
        recent = list(self.nervousness_scores)[-20:]
        
        q1, q3 = np.percentile(recent, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered = [x for x in recent if lower_bound <= x <= upper_bound]
        
        if len(filtered) == 0:
            return np.mean(recent)
        
        weights = np.exp(np.linspace(-1, 0, len(filtered)))
        weighted_avg = np.average(filtered, weights=weights)
        
        return weighted_avg
    
    def _apply_baseline_adjustment(self, score):
        """Adjust score based on calibrated baseline."""
        if self.baseline_movement and len(self.movement_buffer) > 0:
            current_movement = np.mean(list(self.movement_buffer)[-10:])
            if current_movement < self.baseline_movement * 1.5:
                score *= 0.8
        
        return score
    
    def _draw_advanced_overlay(self, frame, score, indicators, confidence):
        """Draw comprehensive information overlay."""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (420, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status header
        status = "NERVOUS" if score > self.threshold else "CALM"
        color = (0, 0, 255) if score > self.threshold else (0, 255, 0)
        
        cv2.putText(frame, f"Status: {status}", (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Score: {score:.3f}", (15, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (15, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Calibration status
        if not self.is_calibrated:
            cv2.putText(frame, f"Calibrating... {self.calibration_frames}/60", 
                       (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Individual indicators
        y_pos = 160
        for name, value in indicators.items():
            label = name.replace('_', ' ').title()[:20]
            
            if value < 0.4:
                bar_color = (0, 255, 0)
            elif value < 0.7:
                bar_color = (0, 165, 255)
            else:
                bar_color = (0, 0, 255)
            
            cv2.putText(frame, f"{label}:", (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            bar_width = int(value * 180)
            cv2.rectangle(frame, (200, y_pos-12), (200 + bar_width, y_pos-2),
                         bar_color, -1)
            cv2.rectangle(frame, (200, y_pos-12), (380, y_pos-2),
                         (100, 100, 100), 1)
            
            cv2.putText(frame, f"{value:.2f}", (385, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            y_pos += 20
    
    def get_most_nervous_capture(self):
            """
            Returns the image with the highest nervousness score.
            Used for report generation.
            """
            if not self.saved_images:
                return None

            # saved_images = [(score, filepath, timestamp)]
            return max(self.saved_images, key=lambda x: x[0])
    
    def save_capture(self, frame, score, indicators=None, confidence=0.0):
        """Save frame with comprehensive metadata and avoid duplicates."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if similar score already exists
        similar_image = self._find_similar_score(score)
        
        if similar_image:
            old_score, old_filepath, old_timestamp = similar_image
            
            # Keep the image with higher score (more nervous)
            if score > old_score:
                # Delete old image and its metadata
                self._delete_image(old_filepath)
                print(f"🗑️  Deleted duplicate: {os.path.basename(old_filepath)} (score: {old_score:.3f})")
                
                # Remove from tracking list
                self.saved_images.remove(similar_image)
            else:
                # Current image is less nervous, don't save it
                print(f"⏭️  Skipped similar image (score: {score:.3f} vs existing {old_score:.3f})")
                return None
        
        # Save new image
        state = "nervous" if score > self.threshold else "calm"
        filename = f"{state}_{timestamp}_s{score:.2f}_c{confidence:.2f}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        cv2.imwrite(filepath, frame)
        
        print("IMAGE SAVED AT:", filepath)
        
        # Save metadata
        if indicators:
            meta_file = filepath.replace('.jpg', '_meta.txt')
            with open(meta_file, 'w') as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"State: {state.upper()}\n")
                f.write(f"Overall Score: {score:.3f}\n")
                f.write(f"Confidence: {confidence:.2f}\n")
                f.write(f"Calibrated: {self.is_calibrated}\n")
                f.write("\nDetailed Indicators:\n")
                for key, value in sorted(indicators.items()):
                    f.write(f"  {key}: {value:.3f}\n")
                
                f.write("\nInterpretation:\n")
                if score < 0.3:
                    f.write("  Very calm and relaxed state\n")
                elif score < 0.5:
                    f.write("  Normal/neutral state\n")
                elif score < 0.7:
                    f.write("  Mild nervousness detected\n")
                else:
                    f.write("  Significant nervousness indicators\n")
        
        # Track this image
        self.saved_images.append((score, filepath, timestamp))
        
        print(f"✓ Saved: {filename} | Score: {score:.3f} | Confidence: {confidence:.2f}")
        return filepath
    
    
    

    def _find_similar_score(self, current_score):
        """Find if an almost identical score already exists in saved images."""
        for saved_score, filepath, timestamp in self.saved_images:
            # Only check for nearly identical scores (difference <= 0.01)
            # Example: 0.71 and 0.72 are different, but 0.710 and 0.711 are same
            if abs(current_score - saved_score) <= self.score_similarity_threshold:
                return (saved_score, filepath, timestamp)
        return None
    
    def _delete_image(self, filepath):
        """Delete image file and its metadata."""
        try:
            # Delete image
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Delete metadata
            meta_file = filepath.replace('.jpg', '_meta.txt')
            if os.path.exists(meta_file):
                os.remove(meta_file)
        except Exception as e:
            print(f"⚠️  Error deleting {filepath}: {e}")


def main():
    """
    Main function - ONLY saves images when NERVOUS (not when calm).
    """
    # Initialize detector
    detector = NervousnessDetector(
        nervousness_threshold=0.65,
        save_dir="nervous_captures",
        buffer_size=8
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("=" * 60)
    print("HIGH-ACCURACY NERVOUSNESS DETECTION SYSTEM")
    print("=" * 60)
    print("Advanced Analysis Methods:")
    print("  • Optical flow tracking (micro-movements)")
    print("  • Statistical pattern analysis")
    print("  • Eye aspect ratio monitoring")
    print("  • Head pose variation")
    print("  • Lip/mouth movement detection")
    print("  • Movement frequency analysis")
    print("  • Adaptive baseline calibration")
    print("\n⚠️  SAVE MODE: NERVOUS ONLY (Calm states ignored)")
    print("\nPress 'q' to quit, 'r' to recalibrate")
    print("=" * 60)
    
    last_save_time = 0
    save_cooldown = 3
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        score, annotated, indicators, confidence = detector.detect_nervousness(frame)
        
        current_time = datetime.now().timestamp()
        
        # ONLY save when NERVOUS (score > threshold) and confident
        if score > detector.threshold and confidence > 0.5:
            if current_time - last_save_time > save_cooldown:
                detector.save_capture(frame, score, indicators, confidence)
                last_save_time = current_time
        
        cv2.imshow('High-Accuracy Nervousness Detector', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.is_calibrated = False
            detector.calibration_frames = 0
            print("\n♻ Recalibrating...")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Detection session ended")


if __name__ == "__main__":
    print("NervousnessDetector module loaded (API mode)")