import cv2
import numpy as np
import winsound
import time
import os
import urllib.request
from collections import deque, defaultdict

class AdvancedFaceMaskDetector:
    def __init__(self):
        try:
            self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
        except:
            print("Face detection models not found. Downloading...")
            self.download_face_detection_models()
            self.face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

        self.confidence_threshold = 0.6
        self.last_alert_time = 0
        self.alert_cooldown = 2

        self.trackers = []
        self.track_id_count = 0

        self.mask_history = defaultdict(lambda: deque(maxlen=15))
        self.mask_vote_threshold = 0.6

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def download_face_detection_models(self):
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        if not os.path.exists('deploy.prototxt'):
            urllib.request.urlretrieve(prototxt_url, 'deploy.prototxt')
        if not os.path.exists('res10_300x300_ssd_iter_140000.caffemodel'):
            urllib.request.urlretrieve(model_url, 'res10_300x300_ssd_iter_140000.caffemodel')

    def adjust_brightness_contrast(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def detect_faces(self, frame):
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
                width, height = x2 - x, y2 - y
                if width > 40 and height > 40:
                    face_boxes.append([x, y, width, height])
        return face_boxes

    def detect_mask(self, face_roi):
        if face_roi.size == 0:
            return False, 0.5

        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        height, width = face_roi.shape[:2]
        lower_face = hsv[height//3:, :]

        mask_ranges = [
            ([100, 50, 50], [130, 255, 255]),
            ([85, 50, 50], [115, 255, 255]),
            ([0, 0, 180], [180, 40, 255]),
            ([0, 0, 0], [180, 255, 60]),
            ([40, 50, 50], [80, 255, 255]),
            ([0, 0, 60], [180, 40, 180]),
            ([15, 100, 100], [35, 255, 255]),
            ([165, 50, 50], [179, 255, 255])
        ]

        total_mask_pixels = 0
        for lower, upper in mask_ranges:
            mask = cv2.inRange(lower_face, np.array(lower), np.array(upper))
            total_mask_pixels += cv2.countNonZero(mask)

        total_pixels = lower_face.shape[0] * lower_face.shape[1]
        mask_ratio = total_mask_pixels / total_pixels if total_pixels > 0 else 0

        blur = cv2.GaussianBlur(lower_face, (5, 5), 0)
        laplacian_var = cv2.Laplacian(blur, cv2.CV_64F).var()
        texture_score = 1 - min(laplacian_var / 50, 1)

        combined_score = mask_ratio * 0.7 + texture_score * 0.3
        has_mask = combined_score > 0.3

        return has_mask, combined_score

    def play_alert(self):
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            try:
                winsound.Beep(800, 300)
                self.last_alert_time = current_time
            except:
                print("⚠️ ALERT: No mask detected!")

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                adjusted_frame = self.adjust_brightness_contrast(frame)
                detections = self.detect_faces(adjusted_frame)

                for (x, y, w, h) in detections:
                    face_roi = adjusted_frame[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        has_mask, confidence = self.detect_mask(face_roi)
                        if has_mask:
                            label = f"Mask ({confidence:.2f})"
                            color = (0, 255, 0)
                        else:
                            label = f"No Mask ({confidence:.2f})"
                            color = (0, 0, 255)
                            self.play_alert()

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.imshow("Advanced Face Mask Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = AdvancedFaceMaskDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")

