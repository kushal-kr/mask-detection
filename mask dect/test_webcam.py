import cv2
import sys

def test_webcam():
    print("Testing webcam connection...")
    
    # Try to access webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam. Trying alternative indices...")
        
        # Try different camera indices
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ Found webcam at index {i}")
                break
        else:
            print("❌ No webcam found. Please check:")
            print("1. Webcam is connected")
            print("2. Webcam permissions are granted")
            print("3. Webcam is not being used by another application")
            return False
    else:
        print("✅ Webcam opened successfully at index 0")
    
    # Test frame capture
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame captured successfully: {frame.shape}")
        print("✅ Webcam is working properly!")
        
        # Show a test frame for 3 seconds
        cv2.imshow("Webcam Test - Press any key to close", frame)
        print("Showing test frame for 3 seconds...")
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        
    else:
        print("❌ Could not capture frame from webcam")
        cap.release()
        return False
    
    cap.release()
    return True

if __name__ == "__main__":
    if test_webcam():
        print("\n🎉 Webcam test passed! You can now run the mask detection system.")
        print("Run: python mask_detection_stable.py")
    else:
        print("\n❌ Webcam test failed. Please fix webcam issues before running mask detection.")
    
    input("Press Enter to exit...")
