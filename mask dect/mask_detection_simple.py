import cv2
import numpy as np
import winsound
import time
import os
import urllib.request

class SimpleFaceMaskDetector:
    def __init__(self):
        # Load face detection model (OpenCV DNN)
        try:
            self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        except:
            print("Face detection models not found. Downloading...")
            self.download_face_detection_models()
            self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        
        self.confidence_threshold = 0.5
        self.last_alert_time = 0
        self.alert_cooldown = 2  # seconds between alerts
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        print("Face Mask Detection System Started")
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
        """Detect faces in the frame using OpenCV DNN"""
        (h, w) = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Pass blob through network
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        face_locations = []
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Compute bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding boxes are within frame dimensions
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                
                face_locations.append((startX, startY, endX, endY))
        
        return face_locations
    
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
        """Main detection loop"""
        try:
            frame_count = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 3rd frame for better performance
                if frame_count % 3 == 0:
                    # Detect faces
                    face_locations = self.detect_faces(frame)
                    
                    # Process each detected face
                    for (startX, startY, endX, endY) in face_locations:
                        # Extract face region
                        face_roi = frame[startY:endY, startX:endX]
                        
                        if face_roi.size > 0:
                            # Detect mask
                            has_mask, confidence = self.detect_mask_advanced(face_roi)
                            
                            # Determine label and color
                            if has_mask:
                                label = f"Mask ({confidence:.2f})"
                                color = (0, 255, 0)  # Green
                                status_text = "✓ MASKED"
                            else:
                                label = f"No Mask ({confidence:.2f})"
                                color = (0, 0, 255)  # Red
                                status_text = "⚠ NO MASK"
                                self.play_alert()  # Trigger audio alert
                            
                            # Draw bounding box and label
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
                            
                            # Draw label background
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(frame, (startX, startY - 35), 
                                        (startX + label_size[0], startY), color, -1)
                            
                            # Draw label text
                            cv2.putText(frame, label, (startX, startY - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Face Mask Detection System", frame)
                
                # Break loop on 'q' key press
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
        print("Face Mask Detection System Stopped")

if __name__ == "__main__":
    try:
        print("=== Face Mask Detection System ===")
        print("Features:")
        print("- Real-time face detection")
        print("- Mask/No-mask classification")
        print("- Audio alerts for unmasked faces")
        print("- Multiple face detection support")
        print("\nStarting system...")
        
        detector = SimpleFaceMaskDetector()
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. A webcam connected")
        print("2. OpenCV installed (pip install opencv-python)")
        print("3. Internet connection for downloading models")
        input("Press Enter to exit...")
