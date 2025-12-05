# Real-Time Face Mask Detection System

A Python-based real-time face mask detection system using OpenCV that activates your webcam to detect whether people are wearing masks or not.

## Features

- ✅ **Real-time detection** via webcam
- ✅ **Multiple face detection** support
- ✅ **Color-coded bounding boxes**:
  - 🟢 Green box for masked faces
  - 🔴 Red box for unmasked faces
- ✅ **Audio alerts** when no mask is detected
- ✅ **No GUI required** - runs directly from terminal
- ✅ **Automatic model download** on first run

## Files Included

1. `mask_detection_simple.py` - Main detection script (recommended)
2. `mask_detection.py` - Advanced version with TensorFlow (requires more dependencies)
3. `requirements_simple.txt` - Basic dependencies
4. `requirements.txt` - Full dependencies for advanced version
5. `setup_and_run.bat` - Windows setup script

## Quick Start (Windows)

### Option 1: Easy Setup (Recommended)
1. Double-click `setup_and_run.bat`
2. The script will automatically install dependencies and run the system

### Option 2: Manual Setup
1. Install Python from https://python.org
2. Open Command Prompt or PowerShell
3. Navigate to this folder
4. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```
5. Run the system:
   ```bash
   python mask_detection_simple.py
   ```

## How It Works

### Detection Process
1. **Face Detection**: Uses OpenCV's DNN module with pre-trained models
2. **Mask Classification**: Combines multiple detection methods:
   - Color analysis in HSV space (detects common mask colors)
   - Edge detection (identifies mask boundaries)
   - Texture analysis (masks have smoother textures)
3. **Alert System**: Plays audio beep when unmasked faces are detected

### Visual Indicators
- **Green bounding box**: Person wearing a mask ✅
- **Red bounding box**: Person not wearing a mask ❌
- **Confidence score**: Shows detection certainty (0.00-1.00)

### Audio Alerts
- Beep sound plays when unmasked face is detected
- 2-second cooldown between alerts to prevent spam
- Fallback text alert if audio fails

## Controls

- **Press 'q'** to quit the application
- **Press 'ESC'** to exit
- **Ctrl+C** in terminal to force stop

## System Requirements

- **Python 3.7+**
- **Webcam** (built-in or external)
- **Internet connection** (for downloading models on first run)
- **Windows** (for audio alerts - can be modified for other OS)

## Dependencies

### Basic Version (mask_detection_simple.py)
- `opencv-python` - Computer vision library
- `numpy` - Numerical computing

### Advanced Version (mask_detection.py)
- All basic dependencies plus:
- `tensorflow` - Deep learning framework

## Troubleshooting

### Common Issues

1. **"Could not open webcam"**
   - Ensure webcam is connected and not used by other applications
   - Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the code

2. **"Face detection models not found"**
   - Ensure internet connection for automatic download
   - Models will be downloaded to the same folder as the script

3. **No audio alerts**
   - Check system volume
   - Audio alerts use Windows `winsound` module
   - Text alerts will appear in console as fallback

4. **Poor detection accuracy**
   - Ensure good lighting conditions
   - Keep face clearly visible to camera
   - Adjust `confidence_threshold` in code if needed

### Performance Tips
- Good lighting improves detection accuracy
- Position face 2-3 feet from camera for best results
- Avoid busy backgrounds for better face detection

## Technical Details

### Models Used
- **Face Detection**: OpenCV DNN with ResNet-10 SSD
- **Mask Detection**: Multi-method approach combining:
  - HSV color space analysis
  - Canny edge detection
  - Laplacian texture analysis

### Detection Algorithm
1. Convert frame to blob format
2. Pass through face detection network
3. Extract face regions above confidence threshold
4. Analyze each face for mask presence using combined scoring
5. Draw bounding boxes and trigger alerts

## Customization

### Adjusting Sensitivity
Edit these values in the script:
```python
self.confidence_threshold = 0.5  # Face detection sensitivity
has_mask = combined_score > 0.25  # Mask detection threshold
self.alert_cooldown = 2  # Seconds between alerts
```

### Adding New Mask Colors
Add color ranges to `mask_ranges` list in `detect_mask_advanced()` method.

## License

This project is for educational and research purposes. Please ensure compliance with privacy laws when using camera-based detection systems.

## Support

If you encounter issues:
1. Check system requirements
2. Verify webcam functionality
3. Ensure stable internet connection
4. Try running with administrator privileges if needed

---

**Note**: This system is designed for demonstration purposes. For production use, consider using professionally trained models and additional validation methods.
