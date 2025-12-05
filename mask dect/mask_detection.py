import cv2
import numpy as np
import winsound
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array
import os

class FaceMaskDetector:
    def __init__(self):
        # Load face detection model (OpenCV DNN)
        self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        
        # Load mask detection model (you'll need to download or train this)
        # For now, we'll create a simple detection based on facial features
        self.confidence_threshold = 0.5
        self.last_alert_time = 0
        self.alert_cooldown = 2  # seconds between alerts
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        print("Face Mask Detection System Started")
        print("Press 'q' to quit")
    
    def detect_faces(self, frame):
        """Detect faces in the frame using OpenCV DNN"""
        (h, w) = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Pass blob through network
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        locs = []
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Compute bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding boxes are within frame dimensions
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                
                # Extract face ROI
                face = frame[startY:endY, startX:endX]
                if face.size > 0:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))
        
        return (locs, faces)
    
    def detect_mask_simple(self, face_roi):
        """Simple mask detection based on color analysis in lower face region"""
        if face_roi.size == 0:
            return False, 0.5
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        # Focus on lower half of face (mouth/nose area)
        height = face_roi.shape[0]
        lower_face = hsv[height//2:, :]
        
        # Define mask color ranges (common mask colors: blue, white, black)
        # Blue masks
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(lower_face, blue_lower, blue_upper)
        
        # White masks
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(lower_face, white_lower, white_upper)
        
        # Black masks
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])
        black_mask = cv2.inRange(lower_face, black_lower, black_upper)
        
        # Combine all mask detections
        combined_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(white_mask, black_mask))
        
        # Calculate percentage of mask pixels
        mask_pixels = cv2.countNonZero(combined_mask)
        total_pixels = lower_face.shape[0] * lower_face.shape[1]
        mask_ratio = mask_pixels / total_pixels if total_pixels > 0 else 0
        
        # If more than 30% of lower face is covered by mask colors, consider it masked
        has_mask = mask_ratio > 0.3
        confidence = mask_ratio
        
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
                print("ALERT: No mask detected!")  # Fallback if sound fails
    
    def run(self):
        """Main detection loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Detect faces
                (locs, faces) = self.detect_faces(frame)
                
                # Loop over detected faces
                for (box, face) in zip(locs, faces):
                    (startX, startY, endX, endY) = box
                    
                    # Extract face region for mask detection
                    face_roi = frame[startY:endY, startX:endX]
                    
                    # Detect mask
                    has_mask, confidence = self.detect_mask_simple(face_roi)
                    
                    # Determine label and color
                    if has_mask:
                        label = f"Mask ({confidence:.2f})"
                        color = (0, 255, 0)  # Green
                    else:
                        label = f"No Mask ({confidence:.2f})"
                        color = (0, 0, 255)  # Red
                        self.play_alert()  # Trigger audio alert
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(frame, label, (startX, startY - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                
                # Display frame
                cv2.imshow("Face Mask Detection", frame)
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
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

def download_face_detection_models():
    """Download required face detection models if they don't exist"""
    import urllib.request
    
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

if __name__ == "__main__":
    try:
        # Download models if needed
        download_face_detection_models()
        
        # Initialize and run detector
        detector = FaceMaskDetector()
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the required dependencies installed.")
