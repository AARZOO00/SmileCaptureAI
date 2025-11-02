# Capture Smile AI

## Overview
**Capture Smile AI** is a Python computer vision application that uses OpenCV and Haar Cascade classifiers to detect smiles in real-time from your webcam and automatically captures photos when you smile!

**Purpose:** Create an interactive, fun application that demonstrates face and smile detection using machine learning.

**Current State:** Fully functional MVP with real-time smile detection and automatic photo capture.

## Recent Changes
- **2025-11-02:** Initial project creation
  - Implemented real-time webcam access using OpenCV
  - Added face detection using Haar Cascade Classifiers
  - Added smile detection with automatic photo capture
  - Created sequential photo saving system (smile_1.jpg, smile_2.jpg, etc.)
  - Added visual feedback with rectangles around faces and smiles
  - Implemented on-screen notification messages
  - Added beginner-friendly code structure with detailed comments

## Features
✅ **Real-time webcam access** - Live video feed from your camera
✅ **Face detection** - Automatically detects faces using Haar Cascades
✅ **Smile detection** - Identifies smiles within detected faces
✅ **Auto photo capture** - Saves photos automatically when you smile
✅ **Visual feedback** - Green rectangles around faces, blue around smiles
✅ **On-screen notifications** - "Smile detected!" message appears after capture
✅ **Smart cooldown system** - Prevents multiple captures of the same smile
✅ **Sequential file naming** - Photos saved as smile_1.jpg, smile_2.jpg, etc.
✅ **Organized storage** - All photos saved in 'captured_smiles' folder

## Project Structure
```
.
├── capture_smile.py      # Main application file
├── captured_smiles/      # Folder where photos are saved (auto-created)
├── replit.md            # Project documentation (this file)
├── .gitignore           # Git ignore rules
└── requirements.txt     # Python dependencies (auto-generated)
```

## How to Use
1. **Run the application:** Click the "Run" button or execute `python capture_smile.py`
2. **Position yourself:** Sit in front of your webcam
3. **Smile!** The app will automatically capture a photo when it detects your smile
4. **Exit:** Press 'q' on your keyboard to quit the application
5. **View photos:** Check the `captured_smiles` folder for your captured images

## Technical Details

### Dependencies
- **Python 3.11** - Programming language
- **OpenCV (cv2)** - Computer vision library for image processing and webcam access
- **NumPy** - Numerical computing library (required by OpenCV)

### Haar Cascade Classifiers
The application uses two pre-trained Haar Cascade XML files:
- `haarcascade_frontalface_default.xml` - Detects frontal faces
- `haarcascade_smile.xml` - Detects smiles

These classifiers come bundled with OpenCV and use machine learning to identify patterns in images.

### How It Works
1. **Camera Initialization:** Opens the default webcam (device 0)
2. **Classifier Loading:** Loads pre-trained Haar Cascade models
3. **Frame Processing Loop:**
   - Captures frame from webcam
   - Converts to grayscale for better detection
   - Detects faces in the frame
   - For each face, searches for smiles
   - Draws rectangles around detected features
   - Captures photo if smile is detected (with cooldown)
   - Displays the processed frame
4. **Photo Saving:** Saves images with sequential numbering
5. **Cleanup:** Releases camera and closes windows on exit

### Key Parameters
- **Face Detection:** `scaleFactor=1.3`, `minNeighbors=5`
- **Smile Detection:** `scaleFactor=1.8`, `minNeighbors=20` (stricter for accuracy)
- **Smile Cooldown:** 60 frames (~2 seconds) to prevent duplicate captures
- **Message Duration:** 30 frames (~1 second) for "Smile detected!" notification

## Code Structure
The code follows a beginner-friendly modular approach:
- `initialize_camera()` - Sets up webcam access
- `load_classifiers()` - Loads Haar Cascade models
- `detect_faces_and_smiles()` - Performs the actual detection
- `save_photo()` - Handles photo saving with sequential naming
- `display_message()` - Shows on-screen notifications
- `main()` - Orchestrates the entire application flow

## Future Enhancements
Potential features for future versions:
- Photo gallery viewer for browsing captured smiles
- Adjustable smile detection sensitivity
- Photo preview overlay after capture
- Timestamp-based filenames
- Session statistics (smiles per minute, total captures, etc.)
- Filters and effects on captured photos
- Multiple face detection and tracking

## Troubleshooting
- **Camera not accessible:** Make sure no other application is using your webcam
- **No faces detected:** Ensure good lighting and face the camera directly
- **Smiles not detected:** Try a bigger, more pronounced smile
- **Application won't start:** Verify OpenCV is installed correctly

## Architecture
This is a standalone Python application using:
- **OpenCV** for computer vision tasks
- **Haar Cascades** for object detection (faces and smiles)
- **File I/O** for photo storage
- **Real-time video processing** at ~30 fps

The application runs in a single thread with a continuous frame processing loop.
