import cv2
import numpy as np
import winsound
import time
import os
import urllib.request
from collections import deque, defaultdict

class KalmanBoxTracker:
    """Kalman filter for tracking bounding boxes"""
    def __init__(self, bbox):
        # State: [x, y, w, h, dx, dy, dw, dh]
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0,0],
                                              [0,1,0,0,0,0,0,0],
                                              [0,0,1,0,0,0,0,0],
                                              [0,0,0,1,0,0,0,0]], np.float32)
        
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0,0],
                                             [0,1,0,0,0,1,0,0],
                                             [0,0,1,0,0,0,1,0],
                                             [0,0,0,1,0,0,0,1],
                                             [0,0,0,0,1,0,0,0],
                                             [0,0,0,0,0,1,0,0],
                                             [0,0,0,0,0,0,1,0],
                                             [0,0,0,0,0,0,0,1]], np.float32)
        
        self.kf.processNoiseCov = 0.03 * np.eye(8, dtype=np.float32)
        self.kf.measurementNoiseCov = 0.1 * np.eye(4, dtype=np.float32)
        
        # Initialize state
        x, y, w, h = bbox
        self.kf.statePre = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        
    def update(self, bbox):
        """Update Kalman filter with new measurement"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        measurement = np.array(bbox, dtype=np.float32)
        self.kf.correct(measurement)
        
    def predict(self):
        """Predict next state"""
        if self.kf.statePost[2] + self.kf.statePost[6] <= 0:
            self.kf.statePost[6] *= 0.0
        if self.kf.statePost[3] + self.kf.statePost[7] <= 0:
            self.kf.statePost[7] *= 0.0
            
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        state = self.kf.statePost
        return [int(state[0]), int(state[1]), int(state[2]), int(state[3])]
    
    def get_state(self):
        """Get current bounding box estimate"""
        state = self.kf.statePost
        return [int(state[0]), int(state[1]), int(state[2]), int(state[3])]

class UltraStableFaceMaskDetector:
    def __init__(self):
        # Load face detection model
        try:
            self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        except:
            print("Face detection models not found. Downloading...")
            self.download_face_detection_models()
            self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        
        # Detection parameters
        self.confidence_threshold = 0.7  # Higher threshold for ultra stability
        self.last_alert_time = 0
        self.alert_cooldown = 3  # Longer cooldown to prevent spam
        
        # Tracking parameters
        self.trackers = []
        self.track_id_count = 0
        self.max_disappeared = 15  # Frames before removing tracker
        self.min_hits = 5  # Minimum hits before confirming track
        
        # Mask detection smoothing
        self.mask_history = defaultdict(lambda: deque(maxlen=15))  # Per-track mask history
        self.mask_vote_threshold = 0.6  # 60% votes needed for mask decision
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 20)  # Slightly higher FPS for smoother tracking
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Ultra Stable Face Mask Detection System Started")
        print("- Advanced Kalman filter tracking")
        print("- Temporal mask detection smoothing")
        print("- Reduced false positives")
        print("Press 'q' to quit")
    
    def download_face_detection_models(self):
        """Download required face detection models"""
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        if not os.path.exists('deploy.prototxt'):
            print("Downloading deploy.prototxt...")
            urllib.request.urlretrieve(prototxt_url, 'deploy.prototxt')
            print("Downloaded deploy.prototxt")
        
        if not os.path.exists('res10_300x300_ssd_iter_140000.caffemodel'):
            print("Downloading face detection model (this may take a while)...")
            urllib.request.urlretrieve(model_url, 'res10_300x300_ssd_iter_140000.caffemodel')
            print("Downloaded face detection model")
    
    def detect_faces(self, frame):
        """Detect faces using DNN"""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        face_boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                x, y = max(0, x), max(0, y)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                
                # Convert to (x, y, width, height) format
                width, height = x2 - x, y2 - y
                if width > 40 and height > 40:  # Minimum size filter
                    face_boxes.append([x, y, width, height])
        
        return face_boxes
    
    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """Associate detections to existing trackers using IoU"""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.compute_iou(det, trk)
        
        # Hungarian algorithm would be ideal here, but using simple greedy matching
        matched_indices = []
        for d in range(len(detections)):
            if len(trackers) == 0:
                break
            best_match = np.argmax(iou_matrix[d])
            if iou_matrix[d, best_match] >= iou_threshold:
                matched_indices.append([d, best_match])
                iou_matrix[:, best_match] = 0  # Remove this tracker from consideration
        
        if len(matched_indices) == 0:
            matched_indices = np.empty((0, 2), dtype=int)
        else:
            matched_indices = np.array(matched_indices)
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if len(matched_indices) == 0 or d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if len(matched_indices) == 0 or t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        return matched_indices, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def compute_iou(self, bbox1, bbox2):
        """Compute Intersection over Union (IoU) of two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection area
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def detect_mask_stable(self, face_roi, track_id):
        """Stable mask detection with voting system"""
        if face_roi.size == 0:
            return False, 0.5
        
        # Current frame mask detection
        current_mask, current_confidence = self.detect_mask_single_frame(face_roi)
        
        # Add to history
        self.mask_history[track_id].append({
            'mask': current_mask,
            'confidence': current_confidence
        })
        
        # If not enough history, return current result
        if len(self.mask_history[track_id]) < 3:
            return current_mask, current_confidence
        
        # Voting system over recent history
        recent_history = list(self.mask_history[track_id])
        mask_votes = sum(1 for h in recent_history if h['mask'])
        total_votes = len(recent_history)
        
        # Calculate average confidence
        avg_confidence = sum(h['confidence'] for h in recent_history) / total_votes
        
        # Decision based on majority vote
        stable_mask = mask_votes / total_votes >= self.mask_vote_threshold
        
        return stable_mask, avg_confidence
    
    def detect_mask_single_frame(self, face_roi):
        """Single frame mask detection using color and additional methods"""
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        height, width = face_roi.shape[:2]
        
        # Focus on lower 2/3 of face
        lower_face = hsv[height//3:, :]
        
        # Enhanced mask color ranges
        mask_ranges = [
            ([100, 50, 50], [130, 255, 255]),  # Blue
            ([85, 50, 50], [115, 255, 255]),   # Light blue
            ([0, 0, 180], [180, 40, 255]),     # White/light
            ([0, 0, 0], [180, 255, 60]),       # Black
            ([40, 50, 50], [80, 255, 255]),    # Green
            ([0, 0, 60], [180, 40, 180]),      # Gray
            ([15, 100, 100], [35, 255, 255]),  # Yellow for variations
            ([165, 50, 50], [179, 255, 255])   # Red for some variations
        ]
        
        total_mask_pixels = 0
        for lower, upper in mask_ranges:
            mask = cv2.inRange(lower_face, np.array(lower), np.array(upper))
            total_mask_pixels += cv2.countNonZero(mask)
        
        total_pixels = lower_face.shape[0] * lower_face.shape[1]
        mask_ratio = total_mask_pixels / total_pixels if total_pixels > 0 else 0
        
        # Enhanced detection with texture analysis
        blur = cv2.GaussianBlur(lower_face, (5, 5), 0)
        laplacian_var = cv2.Laplacian(blur, cv2.CV_64F).var()
        texture_score = 1 - min(laplacian_var / 50, 1)  # Smoother is more likely a mask

        # Combined score with adjusted weights
        combined_score = mask_ratio * 0.7 + texture_score * 0.3
        has_mask = combined_score > 0.3
        
        return has_mask, combined_score
    
    def play_alert(self):
        """Play audio alert for no mask detection"""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            try:
                winsound.Beep(800, 300)  # Shorter, less annoying beep
                self.last_alert_time = current_time
            except:
                print("⚠️ ALERT: No mask detected!")
    
    def run(self):
        """Main detection loop with ultra-stable tracking"""
        try:
            frame_count = 0
            fps_start_time = time.time()
            fps_counter = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                fps_counter += 1
                
                # Detect faces
                detections = self.detect_faces(frame)
                
                # Predict new locations of existing trackers
                predicted_trackers = []
                for tracker in self.trackers:
                    predicted_box = tracker.predict()
                    predicted_trackers.append(predicted_box)
                
                # Associate detections to trackers
                matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
                    detections, predicted_trackers)
                
                # Update matched trackers
                for m in matched:
                    self.trackers[m[1]].update(detections[m[0]])
                
                # Create new trackers for unmatched detections
                for i in unmatched_dets:
                    trk = KalmanBoxTracker(detections[i])
                    trk.id = self.track_id_count
                    self.track_id_count += 1
                    self.trackers.append(trk)
                
                # Remove old trackers
                self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_disappeared]
                
                # Draw results
                for tracker in self.trackers:
                    if tracker.time_since_update <= 1 and tracker.hits >= self.min_hits:
                        x, y, w, h = tracker.get_state()
                        
                        # Extract face region
                        face_roi = frame[y:y+h, x:x+w]
                        
                        if face_roi.size > 0:
                            # Stable mask detection
                            has_mask, confidence = self.detect_mask_stable(face_roi, tracker.id)
                            
                            # Determine display properties
                            if has_mask:
                                label = f"ID:{tracker.id} Mask ({confidence:.2f})"
                                color = (0, 255, 0)  # Green
                                box_thickness = 2
                            else:
                                label = f"ID:{tracker.id} No Mask ({confidence:.2f})"
                                color = (0, 0, 255)  # Red
                                box_thickness = 3
                                self.play_alert()
                            
                            # Draw ultra-stable bounding box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thickness)
                            
                            # Draw label with background
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 10, y), color, -1)
                            cv2.putText(frame, label, (x + 5, y - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Draw track ID
                            cv2.putText(frame, f"Track: {tracker.hits}", (x, y + h + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Calculate FPS
                if fps_counter >= 30:
                    fps_end_time = time.time()
                    fps = fps_counter / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    fps_counter = 0
                
                # Display info
                info_text = [
                    f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --",
                    f"Active Tracks: {len([t for t in self.trackers if t.hits >= self.min_hits])}",
                    f"Total Detections: {len(detections)}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(frame, text, (10, 30 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Ultra Stable Face Mask Detection", frame)
                
                # Exit condition
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Ultra Stable Face Mask Detection System Stopped")

if __name__ == "__main__":
    try:
        print("=== Ultra Stable Face Mask Detection System ===")
        print("Advanced Features:")
        print("✓ Kalman filter tracking for ultra-smooth bounding boxes")
        print("✓ Track ID assignment and persistence")
        print("✓ Temporal mask detection with voting system")
        print("✓ Reduced false positives and blinking")
        print("✓ Performance optimizations")
        print("\nInitializing system...")
        
        detector = UltraStableFaceMaskDetector()
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Troubleshooting:")
        print("1. Ensure webcam is connected and not in use")
        print("2. Check OpenCV installation: pip install opencv-python")
        print("3. Verify internet connection for model download")
        input("Press Enter to exit...")
