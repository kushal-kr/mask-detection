import cv2
import numpy as np
import winsound
import time
import os
import urllib.request
from collections import deque, defaultdict

class DimLightOptimizedMaskDetector:
    def __init__(self):
        """Initialize the detector with dim light optimizations"""
        try:
            self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        except:
            print("Face detection models not found. Downloading...")
            self.download_face_detection_models()
            self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

        # Detection parameters optimized for dim light
        self.confidence_threshold = 0.5  # Lower threshold for dim light
        self.last_alert_time = 0
        self.alert_cooldown = 2

        # Enhanced tracking for stability
        self.mask_history = defaultdict(lambda: deque(maxlen=20))  # Longer history
        self.detection_history = deque(maxlen=10)
        self.face_trackers = {}
        self.tracker_id = 0

        # Lighting adaptation parameters
        self.brightness_threshold = 80  # Threshold to detect dim lighting
        self.current_brightness = 128   # Current frame brightness estimate
        self.brightness_history = deque(maxlen=30)

        # Initialize webcam with optimal settings
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")

        # Optimize camera settings for low light
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for better exposure
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Try to set exposure and gain for better low light performance
        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # Lower exposure for less noise
            self.cap.set(cv2.CAP_PROP_GAIN, 50)      # Moderate gain
        except:
            pass  # Camera might not support these settings

        print("🌙 Dim Light Optimized Face Mask Detection System Started")
        print("✓ Enhanced brightness and contrast adjustment")
        print("✓ Adaptive lighting detection")
        print("✓ Multi-scale mask detection")
        print("✓ Noise reduction algorithms")
        print("✓ Improved color space analysis")
        print("Press 'q' to quit")

    def download_face_detection_models(self):
        """Download required face detection models"""
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        if not os.path.exists('deploy.prototxt'):
            print("Downloading deploy.prototxt...")
            urllib.request.urlretrieve(prototxt_url, 'deploy.prototxt')
        
        if not os.path.exists('res10_300x300_ssd_iter_140000.caffemodel'):
            print("Downloading face detection model...")
            urllib.request.urlretrieve(model_url, 'res10_300x300_ssd_iter_140000.caffemodel')

    def estimate_brightness(self, frame):
        """Estimate the overall brightness of the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        self.brightness_history.append(brightness)
        
        # Use moving average for stable brightness estimation
        if len(self.brightness_history) > 5:
            self.current_brightness = np.mean(list(self.brightness_history)[-5:])
        else:
            self.current_brightness = brightness
            
        return self.current_brightness

    def enhance_for_dim_light(self, frame):
        """Comprehensive image enhancement for dim lighting conditions"""
        enhanced = frame.copy()
        
        # Method 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Method 2: Gamma correction for brightness adjustment
        if self.current_brightness < self.brightness_threshold:
            gamma = 1.5 if self.current_brightness < 60 else 1.2
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
        
        # Method 3: Bilateral filter for noise reduction while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Method 4: Unsharp masking for detail enhancement
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return enhanced

    def detect_faces_multi_scale(self, frame):
        """Multi-scale face detection for better accuracy in dim light"""
        (h, w) = frame.shape[:2]
        face_boxes = []
        
        # Multiple scales for better detection
        scales = [(300, 300), (416, 416), (512, 512)]
        
        for scale in scales:
            blob = cv2.dnn.blobFromImage(frame, 1.0, scale, (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Adaptive confidence threshold based on lighting
                adaptive_threshold = self.confidence_threshold
                if self.current_brightness < self.brightness_threshold:
                    adaptive_threshold *= 0.8  # Lower threshold for dim light
                
                if confidence > adaptive_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x2, y2) = box.astype("int")
                    x, y = max(0, x), max(0, y)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    width, height = x2 - x, y2 - y
                    
                    if width > 30 and height > 30:  # Smaller minimum size for dim light
                        face_boxes.append({
                            'box': [x, y, width, height], 
                            'confidence': confidence,
                            'scale': scale
                        })
        
        # Remove duplicate detections
        face_boxes = self.remove_duplicate_detections(face_boxes)
        return face_boxes

    def remove_duplicate_detections(self, detections):
        """Remove overlapping detections"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for detection in detections:
            is_duplicate = False
            for filtered_det in filtered:
                if self.calculate_overlap(detection['box'], filtered_det['box']) > 0.3:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered

    def calculate_overlap(self, box1, box2):
        """Calculate overlap ratio between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    def detect_mask_enhanced(self, face_roi, track_id=None):
        """Enhanced mask detection optimized for dim lighting"""
        if face_roi.size == 0:
            return False, 0.5

        # Preprocess face ROI for better mask detection
        enhanced_roi = self.enhance_face_roi(face_roi)
        
        # Method 1: Enhanced HSV analysis
        mask_score_hsv = self.detect_mask_hsv_enhanced(enhanced_roi)
        
        # Method 2: LAB color space analysis (better for lighting variations)
        mask_score_lab = self.detect_mask_lab(enhanced_roi)
        
        # Method 3: Edge and texture analysis
        mask_score_texture = self.detect_mask_texture(enhanced_roi)
        
        # Method 4: Geometric analysis
        mask_score_geometry = self.detect_mask_geometry(enhanced_roi)
        
        # Combine scores with weights optimized for dim light
        combined_score = (
            mask_score_hsv * 0.3 +
            mask_score_lab * 0.3 +
            mask_score_texture * 0.25 +
            mask_score_geometry * 0.15
        )
        
        # Apply temporal smoothing if tracking is available
        if track_id is not None:
            combined_score = self.apply_temporal_smoothing(combined_score, track_id)
        
        # Adaptive threshold based on lighting conditions
        mask_threshold = 0.35 if self.current_brightness < self.brightness_threshold else 0.4
        has_mask = combined_score > mask_threshold
        
        return has_mask, combined_score

    def enhance_face_roi(self, face_roi):
        """Enhance face ROI specifically for mask detection"""
        # Apply local contrast enhancement
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE with smaller tile size for face
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Slight sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced

    def detect_mask_hsv_enhanced(self, face_roi):
        """Enhanced HSV-based mask detection with expanded ranges for dim light"""
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        height, width = face_roi.shape[:2]
        
        # Focus on mouth and nose area (lower 2/3 of face)
        lower_face = hsv[height//3:, :]
        
        # Expanded mask color ranges for dim lighting
        mask_ranges = [
            # Blues (surgical masks)
            ([90, 30, 30], [130, 255, 255]),
            ([100, 20, 20], [140, 255, 255]),
            
            # Whites and light colors
            ([0, 0, 150], [180, 50, 255]),
            ([0, 0, 120], [180, 30, 255]),
            
            # Blacks and dark colors
            ([0, 0, 0], [180, 255, 80]),
            ([0, 0, 0], [180, 50, 60]),
            
            # Greens (some fabric masks)
            ([35, 30, 30], [85, 255, 255]),
            
            # Grays
            ([0, 0, 50], [180, 50, 200]),
            
            # Expanded ranges for cloth masks
            ([0, 10, 80], [180, 60, 220]),
            ([0, 5, 100], [180, 40, 180])
        ]
        
        total_mask_pixels = 0
        total_pixels = lower_face.shape[0] * lower_face.shape[1]
        
        for lower, upper in mask_ranges:
            mask = cv2.inRange(lower_face, np.array(lower), np.array(upper))
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            total_mask_pixels += cv2.countNonZero(mask)
        
        return min(total_mask_pixels / total_pixels, 1.0) if total_pixels > 0 else 0

    def detect_mask_lab(self, face_roi):
        """LAB color space analysis for mask detection (better lighting invariance)"""
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        height, width = face_roi.shape[:2]
        lower_face = lab[height//3:, :]
        
        l, a, b = cv2.split(lower_face)
        
        # Detect uniform regions in LAB space (masks tend to be more uniform)
        l_std = np.std(l)
        a_std = np.std(a)
        b_std = np.std(b)
        
        # Lower standard deviation indicates more uniform color (likely a mask)
        uniformity_score = 1.0 - min((l_std + a_std + b_std) / 150.0, 1.0)
        
        # Detect specific LAB ranges that indicate masks
        mask_pixels = 0
        total_pixels = lower_face.shape[0] * lower_face.shape[1]
        
        # White/light masks in LAB
        white_mask = cv2.inRange(lower_face, np.array([120, 0, 0]), np.array([255, 30, 30]))
        mask_pixels += cv2.countNonZero(white_mask)
        
        # Dark masks in LAB
        dark_mask = cv2.inRange(lower_face, np.array([0, 0, 0]), np.array([80, 30, 30]))
        mask_pixels += cv2.countNonZero(dark_mask)
        
        color_score = mask_pixels / total_pixels if total_pixels > 0 else 0
        
        return (uniformity_score * 0.6 + color_score * 0.4)

    def detect_mask_texture(self, face_roi):
        """Texture analysis for mask detection"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        lower_face = gray[height//3:, :]
        
        # Calculate texture features
        # 1. Local Binary Pattern approximation
        lbp_like = cv2.Laplacian(lower_face, cv2.CV_64F)
        texture_variance = np.var(lbp_like)
        
        # 2. Edge density
        edges = cv2.Canny(lower_face, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 3. Gradient magnitude
        grad_x = cv2.Sobel(lower_face, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(lower_face, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(grad_magnitude)
        
        # Masks typically have lower texture variance and edge density
        texture_score = 1.0 - min(texture_variance / 100.0, 1.0)
        edge_score = 1.0 - min(edge_density * 10, 1.0)
        gradient_score = 1.0 - min(avg_gradient / 50.0, 1.0)
        
        return (texture_score * 0.4 + edge_score * 0.3 + gradient_score * 0.3)

    def detect_mask_geometry(self, face_roi):
        """Geometric analysis to detect mask-like structures"""
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Focus on mouth area
        mouth_region = gray[int(height*0.6):int(height*0.9), int(width*0.2):int(width*0.8)]
        
        if mouth_region.size == 0:
            return 0.0
        
        # Look for horizontal structures (mask edges)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        horizontal_lines = cv2.morphologyEx(mouth_region, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Score based on horizontal line strength
        horizontal_score = np.sum(horizontal_lines > 0) / (mouth_region.shape[0] * mouth_region.shape[1])
        
        return min(horizontal_score * 5, 1.0)

    def apply_temporal_smoothing(self, current_score, track_id):
        """Apply temporal smoothing to reduce flickering"""
        self.mask_history[track_id].append(current_score)
        
        if len(self.mask_history[track_id]) < 3:
            return current_score
        
        # Use weighted average with more weight on recent frames
        history = list(self.mask_history[track_id])
        weights = np.linspace(0.5, 1.0, len(history))
        smoothed_score = np.average(history, weights=weights)
        
        return smoothed_score

    def play_alert(self):
        """Play audio alert for no mask detection"""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            try:
                winsound.Beep(800, 300)
                self.last_alert_time = current_time
            except:
                print("⚠️ ALERT: No mask detected!")

    def run(self):
        """Main detection loop with dim light optimizations"""
        try:
            fps_counter = 0
            fps_start_time = time.time()
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                fps_counter += 1
                
                # Estimate current lighting conditions
                brightness = self.estimate_brightness(frame)
                is_dim_light = brightness < self.brightness_threshold
                
                # Apply comprehensive enhancement for dim lighting
                if is_dim_light:
                    enhanced_frame = self.enhance_for_dim_light(frame)
                else:
                    enhanced_frame = frame.copy()
                
                # Multi-scale face detection
                detections = self.detect_faces_multi_scale(enhanced_frame)
                
                # Process each detection
                for i, detection in enumerate(detections):
                    box = detection['box']
                    confidence = detection['confidence']
                    x, y, w, h = box
                    
                    # Extract face region
                    face_roi = enhanced_frame[y:y+h, x:x+w]
                    
                    if face_roi.size > 0:
                        # Enhanced mask detection with tracking ID
                        has_mask, mask_confidence = self.detect_mask_enhanced(face_roi, track_id=i)
                        
                        # Determine display properties
                        if has_mask:
                            label = f"Mask ({mask_confidence:.2f})"
                            color = (0, 255, 0)  # Green
                            thickness = 2
                        else:
                            label = f"No Mask ({mask_confidence:.2f})"
                            color = (0, 0, 255)  # Red
                            thickness = 3
                            self.play_alert()
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                        
                        # Draw label with background for better visibility
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 10, y), color, -1)
                        cv2.putText(frame, label, (x + 5, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Additional info for debugging
                        cv2.putText(frame, f"Face Conf: {confidence:.2f}", (x, y + h + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Display lighting information
                lighting_status = f"Lighting: {'DIM' if is_dim_light else 'NORMAL'} ({brightness:.0f})"
                lighting_color = (0, 255, 255) if is_dim_light else (255, 255, 255)
                cv2.putText(frame, lighting_status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, lighting_color, 2)
                
                # Display FPS
                if fps_counter >= 30:
                    fps_end_time = time.time()
                    fps = fps_counter / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    fps_counter = 0
                
                if 'fps' in locals():
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Display frame
                cv2.imshow("🌙 Dim Light Optimized Face Mask Detection", frame)
                
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
        print("🌙 Dim Light Optimized Face Mask Detection System Stopped")

if __name__ == "__main__":
    try:
        print("=== 🌙 Dim Light Optimized Face Mask Detection System ===")
        print("Advanced Features for Low-Light Conditions:")
        print("✓ Adaptive brightness and contrast enhancement")
        print("✓ CLAHE (Contrast Limited Adaptive Histogram Equalization)")
        print("✓ Multi-scale face detection")
        print("✓ Enhanced color space analysis (HSV + LAB)")
        print("✓ Texture and geometric analysis")
        print("✓ Temporal smoothing for stability")
        print("✓ Automatic lighting condition detection")
        print("✓ Noise reduction algorithms")
        print("\\nInitializing optimized system...")
        
        detector = DimLightOptimizedMaskDetector()
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Troubleshooting:")
        print("1. Ensure webcam is connected and not in use by other applications")
        print("2. Check OpenCV installation: pip install opencv-python")
        print("3. Verify internet connection for model download")
        print("4. Try adjusting room lighting if detection is still poor")
        input("Press Enter to exit...")
