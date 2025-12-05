import cv2
import numpy as np
import winsound
import time
import os
import urllib.request
from collections import deque

class StableFaceMaskDetector:
    def __init__(self):
        # Load face detection model (OpenCV DNN)
        try:
            self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        except:
            print("Face detection models not found. Downloading...")
            self.download_face_detection_models()
            self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        
        # Detection parameters
        self.confidence_threshold = 0.6  # Increased for more stable detection
        self.last_alert_time = 0
        self.alert_cooldown = 2  # seconds between alerts
        
        # Stability parameters
        self.detection_history = deque(maxlen=5)  # Store last 5 detections for smoothing
        self.mask_history = deque(maxlen=7)  # Store mask detection history
        self.stable_boxes = {}  # Track stable bounding boxes
        self.box_smoothing_factor = 0.7  # How much to smooth box positions
        self.min_detection_frames = 3  # Minimum frames before showing detection
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        # Set camera properties for better stability
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS for stability
        
        print("Stable Face Mask Detection System Started")
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
    
    def detect_faces_stable(self, frame):
        """Detect faces with stability improvements"""
        (h, w) = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Pass blob through network
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        current_detections = []
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Compute bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding boxes are within frame dimensions
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                
                # Only include if box is reasonable size
                box_width = endX - startX
                box_height = endY - startY
                if box_width > 30 and box_height > 30:
                    current_detections.append({
                        'box': (startX, startY, endX, endY),
                        'confidence': confidence,
                        'center': ((startX + endX) // 2, (startY + endY) // 2)
                    })
        
        # Add to detection history
        self.detection_history.append(current_detections)
        
        return self.get_stable_detections()
    
    def get_stable_detections(self):
        """Get stable detections by averaging recent detections"""
        if len(self.detection_history) < self.min_detection_frames:
            return []
        
        # Find consistent detections across frames
        stable_detections = []
        
        # Get the most recent detection as reference
        recent_detections = self.detection_history[-1]
        
        for detection in recent_detections:
            center = detection['center']
            confidence_sum = detection['confidence']
            box_sum = np.array(detection['box'])
            consistent_count = 1
            
            # Check consistency across previous frames
            for history_frame in list(self.detection_history)[-self.min_detection_frames:-1]:
                for hist_detection in history_frame:
                    hist_center = hist_detection['center']
                    # Check if centers are close (within 50 pixels)
                    distance = np.sqrt((center[0] - hist_center[0])**2 + (center[1] - hist_center[1])**2)
                    if distance < 50:
                        confidence_sum += hist_detection['confidence']
                        box_sum += np.array(hist_detection['box'])
                        consistent_count += 1
                        break
            
            # Only include if detected in at least min_detection_frames
            if consistent_count >= self.min_detection_frames:
                # Average the boxes for stability
                avg_box = (box_sum / consistent_count).astype(int)
                avg_confidence = confidence_sum / consistent_count
                
                stable_detections.append({
                    'box': tuple(avg_box),
                    'confidence': avg_confidence
                })
        
        return stable_detections
    
    def smooth_bounding_box(self, new_box, face_id):
        """Smooth bounding box transitions"""
        if face_id not in self.stable_boxes:
            self.stable_boxes[face_id] = new_box
            return new_box
        
        old_box = self.stable_boxes[face_id]
        
        # Smooth each coordinate
        smoothed_box = []
        for i in range(4):
            smoothed_coord = int(old_box[i] * self.box_smoothing_factor + 
                               new_box[i] * (1 - self.box_smoothing_factor))
            smoothed_box.append(smoothed_coord)
        
        smoothed_box = tuple(smoothed_box)
        self.stable_boxes[face_id] = smoothed_box
        return smoothed_box
    
    def detect_mask_stable(self, face_roi):
        """Stable mask detection with temporal smoothing"""
        if face_roi.size == 0:
            return False, 0.5
        
        # Get current mask detection
        current_mask, current_confidence = self.detect_mask_advanced(face_roi)
        
        # Add to mask history
        self.mask_history.append({'mask': current_mask, 'confidence': current_confidence})
        
        # If we don't have enough history, return current result
        if len(self.mask_history) < 3:
            return current_mask, current_confidence
        
        # Count recent mask detections
        recent_history = list(self.mask_history)[-5:]  # Last 5 detections
        mask_count = sum(1 for h in recent_history if h['mask'])
        avg_confidence = sum(h['confidence'] for h in recent_history) / len(recent_history)
        
        # Stabilized decision: need majority vote
        stable_mask = mask_count >= len(recent_history) / 2
        
        return stable_mask, avg_confidence
    
    def detect_mask_advanced(self, face_roi):
        """Advanced mask detection using multiple techniques"""
        if face_roi.size == 0:
            return False, 0.5
        
        # Method 1: Color-based detection in HSV space
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        height, width = face_roi.shape[:2]
        
        # Focus on lower 2/3 of face (mouth/nose area)
        lower_face = hsv[height//3:, :]
        
        # Define comprehensive mask color ranges
        mask_ranges = [
            # Blue masks (surgical masks)
            ([100, 50, 50], [130, 255, 255]),
            # Light blue masks
            ([85, 50, 50], [115, 255, 255]),
            # White/light colored masks
            ([0, 0, 180], [180, 40, 255]),
            # Black masks
            ([0, 0, 0], [180, 255, 60]),
            # Green masks
            ([40, 50, 50], [80, 255, 255]),
            # Gray masks
            ([0, 0, 60], [180, 40, 180])
        ]
        
        total_mask_pixels = 0
        for lower, upper in mask_ranges:
            mask = cv2.inRange(lower_face, np.array(lower), np.array(upper))
            total_mask_pixels += cv2.countNonZero(mask)
        
        total_pixels = lower_face.shape[0] * lower_face.shape[1]
        color_ratio = total_mask_pixels / total_pixels if total_pixels > 0 else 0
        
        # Method 2: Edge detection for mask boundaries
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        lower_gray = gray_face[height//3:, :]
        edges = cv2.Canny(lower_gray, 50, 150)
        
        # Look for horizontal edges (typical mask pattern)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        edge_pixels = cv2.countNonZero(horizontal_edges)
        edge_ratio = edge_pixels / (lower_gray.shape[0] * lower_gray.shape[1])
        
        # Method 3: Texture analysis (masks tend to be smoother)
        face_blur = cv2.GaussianBlur(lower_gray, (5, 5), 0)
        laplacian_var = cv2.Laplacian(face_blur, cv2.CV_64F).var()
        texture_score = 1 - min(laplacian_var / 100, 1)  # Lower variance = smoother = more likely mask
        
        # Combine all methods
        color_weight = 0.6
        edge_weight = 0.3
        texture_weight = 0.1
        
        combined_score = (color_ratio * color_weight + 
                         edge_ratio * edge_weight + 
                         texture_score * texture_weight)
        
        # Decision threshold
        has_mask = combined_score > 0.25
        confidence = min(combined_score * 2, 1.0)  # Scale confidence to 0-1
        
        return has_mask, confidence
    
    def play_alert(self):
        """Play audio alert for no mask detection"""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            try:
                # Play beep sound (frequency, duration in ms)
                winsound.Beep(1000, 500)  # 1000Hz for 500ms
                self.last_alert_time = current_time
            except:
                print("🚨 ALERT: No mask detected!")  # Fallback if sound fails
    
    def run(self):
        """Main detection loop with stability improvements"""
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
                
                # Process every 2nd frame for better performance and stability
                if frame_count % 2 == 0:
                    # Detect faces with stability
                    stable_detections = self.detect_faces_stable(frame)
                    
                    # Process each stable detection
                    for idx, detection in enumerate(stable_detections):
                        box = detection['box']
                        confidence = detection['confidence']
                        
                        # Smooth the bounding box
                        smoothed_box = self.smooth_bounding_box(box, idx)
                        (startX, startY, endX, endY) = smoothed_box
                        
                        # Extract face region
                        face_roi = frame[startY:endY, startX:endX]
                        
                        if face_roi.size > 0:
                            # Stable mask detection
                            has_mask, mask_confidence = self.detect_mask_stable(face_roi)
                            
                            # Determine label and color
                            if has_mask:
                                label = f"Mask ({mask_confidence:.2f})"
                                color = (0, 255, 0)  # Green
                                status_text = "✓ PROTECTED"
                            else:
                                label = f"No Mask ({mask_confidence:.2f})"
                                color = (0, 0, 255)  # Red
                                status_text = "⚠ UNPROTECTED"
                                self.play_alert()  # Trigger audio alert
                            
                            # Draw stable bounding box with thicker lines
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                            
                            # Draw label background for better visibility
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame, (startX, startY - 40), 
                                        (startX + label_size[0] + 10, startY), color, -1)
                            
                            # Draw label text
                            cv2.putText(frame, label, (startX + 5, startY - 15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Draw confidence indicator
                            conf_text = f"Conf: {confidence:.2f}"
                            cv2.putText(frame, conf_text, (startX, endY + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Calculate and display FPS
                if fps_counter >= 30:  # Update FPS every 30 frames
                    fps_end_time = time.time()
                    fps = fps_counter / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    fps_counter = 0
                
                # Add FPS display
                cv2.putText(frame, f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add instructions
                cv2.putText(frame, "Press 'q' or ESC to quit", (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow("Stable Face Mask Detection System", frame)
                
                # Break loop on 'q' key press or ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Stable Face Mask Detection System Stopped")

if __name__ == "__main__":
    try:
        print("=== Stable Face Mask Detection System ===")
        print("Features:")
        print("- Stable bounding boxes with temporal smoothing")
        print("- Reduced blinking and jitter")
        print("- Improved confidence filtering")
        print("- Real-time FPS display")
        print("- Audio alerts for unmasked faces")
        print("\nStarting stable detection system...")
        
        detector = StableFaceMaskDetector()
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. A webcam connected")
        print("2. OpenCV installed (pip install opencv-python)")
        print("3. Internet connection for downloading models")
        input("Press Enter to exit...")
