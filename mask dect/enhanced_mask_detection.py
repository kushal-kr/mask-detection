import cv2
import numpy as np
import winsound
import time
import os
import urllib.request
from collections import deque

class EnhancedStableFaceMaskDetector:
    def __init__(self):
        try:
            self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        except:
            print("Face detection models not found. Downloading...")
            self.download_face_detection_models()
            self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

        self.confidence_threshold = 0.65
        self.last_alert_time = 0
        self.alert_cooldown = 2

        self.detection_history = deque(maxlen=10)
        self.mask_history = deque(maxlen=10)
        self.stable_boxes = {}
        self.box_smoothing_factor = 0.8
        self.min_detection_frames = 3

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 15)

        print("Enhanced Stable Face Mask Detection System Started")
        print("Press 'q' to quit")

    def download_face_detection_models(self):
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
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        current_detections = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                if endX - startX > 30 and endY - startY > 30:
                    current_detections.append({
                        'box': (startX, startY, endX, endY),
                        'confidence': confidence
                    })

        self.detection_history.append(current_detections)
        return self.get_stable_detections()

    def get_stable_detections(self):
        if len(self.detection_history) < self.min_detection_frames:
            return []

        stable_detections = []
        recent_detections = self.detection_history[-1]
        for detection in recent_detections:
            box_sum = np.array(detection['box'])
            consistent_count = 1
            for history_frame in list(self.detection_history)[-self.min_detection_frames:-1]:
                for hist_detection in history_frame:
                    distance = np.linalg.norm(np.array(detection['box'][0:2]) - np.array(hist_detection['box'][0:2]))
                    if distance < 50:
                        box_sum += np.array(hist_detection['box'])
                        consistent_count += 1
                        break
            if consistent_count >= self.min_detection_frames:
                avg_box = (box_sum / consistent_count).astype(int)
                stable_detections.append({
                    'box': tuple(avg_box),
                    'confidence': detection['confidence']
                })

        return stable_detections

    def smooth_bounding_box(self, new_box, face_id):
        if face_id not in self.stable_boxes:
            self.stable_boxes[face_id] = new_box
            return new_box

        old_box = self.stable_boxes[face_id]
        smoothed_box = []
        for i in range(4):
            smoothed_coord = int(old_box[i] * self.box_smoothing_factor + new_box[i] * (1 - self.box_smoothing_factor))
            smoothed_box.append(smoothed_coord)

        smoothed_box = tuple(smoothed_box)
        self.stable_boxes[face_id] = smoothed_box
        return smoothed_box

    def detect_mask(self, face_roi):
        if face_roi.size == 0:
            return False, 0.5

        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        height, width = face_roi.shape[:2]
        lower_face = hsv[height//3:, :]
        blue_mask = cv2.inRange(lower_face, np.array([100, 50, 50]), np.array([130, 255, 255]))
        light_blue_mask = cv2.inRange(lower_face, np.array([85, 50, 50]), np.array([115, 255, 255]))
        white_mask = cv2.inRange(lower_face, np.array([0, 0, 180]), np.array([180, 40, 255]))
        black_mask = cv2.inRange(lower_face, np.array([0, 0, 0]), np.array([180, 255, 60]))
        green_mask = cv2.inRange(lower_face, np.array([40, 50, 50]), np.array([80, 255, 255]))
        gray_mask = cv2.inRange(lower_face, np.array([0, 0, 60]), np.array([180, 40, 180]))

        combined_mask = cv2.bitwise_or(blue_mask,
                                       cv2.bitwise_or(light_blue_mask,
                                       cv2.bitwise_or(white_mask,
                                       cv2.bitwise_or(black_mask,
                                       cv2.bitwise_or(green_mask, gray_mask)))))

        mask_pixels = cv2.countNonZero(combined_mask)
        total_pixels = lower_face.shape[0] * lower_face.shape[1]
        mask_ratio = mask_pixels / total_pixels if total_pixels > 0 else 0

        has_mask = mask_ratio > 0.3
        confidence = mask_ratio

        return has_mask, confidence

    def play_alert(self):
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            try:
                winsound.Beep(1000, 500)
                self.last_alert_time = current_time
            except:
                print("ALERT: No mask detected!")

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                stable_detections = self.detect_faces_stable(frame)
                face_id = 0
                for detection in stable_detections:
                    box = detection['box']
                    (startX, startY, endX, endY) = self.smooth_bounding_box(box, face_id)
                    face_id += 1
                    face_roi = frame[startY:endY, startX:endX]
                    if face_roi.size > 0:
                        has_mask, confidence = self.detect_mask(face_roi)
                        if has_mask:
                            label = f"Mask ({confidence:.2f})"
                            color = (0, 255, 0)
                        else:
                            label = f"No Mask ({confidence:.2f})"
                            color = (0, 0, 255)
                            self.play_alert()

                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                        cv2.putText(frame, label, (startX, startY - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.imshow("Enhanced Stable Face Mask Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("Enhanced Stable Face Mask Detection System Stopped")

if __name__ == "__main__":
    try:
        detector = EnhancedStableFaceMaskDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")

