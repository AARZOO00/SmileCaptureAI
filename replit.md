# Capture Smile AI

## Overview
**Capture Smile AI** is a Python computer vision application that uses OpenCV and Haar Cascade classifiers to detect smiles in real-time from your webcam and automatically captures photos when you smile!

**Purpose:** Create an interactive, fun application that demonstrates face and smile detection using machine learning.

**Current State:** Fully functional MVP with real-time smile detection and automatic photo capture.

## Recent Changes
- **2025-11-02:** Added attractive UI/UX with professional design
  - Modern gradient header bar with app title and photo counter
  - Dynamic footer bar with real-time status updates and instructions
  - Rounded rectangle borders around detected faces for polished look
  - Animated countdown with pulse effect and color changes (green→yellow→red)
  - Glowing circle background for countdown numbers
  - Beautiful success message with fade effects and checkmark icon
  - Stylish face/smile detection with multi-layered borders and labels
  - Enhanced visual feedback with emoji status indicators
  - Professional color scheme throughout the application
  
- **2025-11-02:** Added countdown timer feature
  - Implemented 3-2-1 countdown before photo capture
  - Large centered countdown display for better user experience
  - Updated cooldown timing to accommodate countdown sequence
  
- **2025-11-02:** Initial project creation
  - Implemented real-time webcam access using OpenCV
  - Added face detection using Haar Cascade Classifiers
  - Added smile detection with automatic photo capture
  - Created sequential photo saving system (smile_1.jpg, smile_2.jpg, etc.)
  - Added visual feedback with rectangles around faces and smiles
  - Implemented on-screen notification messages
  - Added beginner-friendly code structure with detailed comments

## Features
✅ **Professional UI/UX Design** - Beautiful modern interface with gradient bars and smooth animations
✅ **Real-time webcam access** - Live video feed from your camera
✅ **Smart face detection** - Automatically detects faces with stylish rounded borders and labels
✅ **Smile detection** - Identifies smiles with multi-layered pink highlighting and emoji indicators
✅ **Animated countdown timer** - Pulsing 3-2-1 countdown with color-coded numbers and circle effects
✅ **Auto photo capture** - Saves photos automatically after countdown
✅ **Header bar** - Displays app title and running photo counter
✅ **Footer bar** - Shows real-time status with emoji indicators and instructions
✅ **Success notifications** - Animated "Photo Captured!" message with fade effects and checkmark
✅ **Dynamic status updates** - Real-time feedback (Looking for faces, Face detected, Smile detected, Countdown)
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
3. **Smile!** The app will detect your smile and start a countdown
4. **Hold your smile:** A large 3-2-1 countdown will appear on screen
5. **Photo captured!** After the countdown, your photo is automatically saved
6. **Exit:** Press 'q' on your keyboard to quit the application
7. **View photos:** Check the `captured_smiles` folder for your captured images

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
   - When smile detected, starts 3-2-1 countdown
   - Displays large countdown numbers on screen
   - Captures photo after countdown completes
   - Displays the processed frame
4. **Photo Saving:** Saves images with sequential numbering
5. **Cleanup:** Releases camera and closes windows on exit

### Key Parameters
- **Face Detection:** `scaleFactor=1.3`, `minNeighbors=5`
- **Smile Detection:** `scaleFactor=1.8`, `minNeighbors=20` (stricter for accuracy)
- **Countdown Duration:** 30 frames per number (~1 second each for 3, 2, 1)
- **Smile Cooldown:** 90 frames (~3 seconds) to prevent duplicate captures
- **Message Duration:** 30 frames (~1 second) for "Smile detected!" notification

## Code Structure
The code follows a beginner-friendly modular approach with beautiful UI components:

**UI/UX Functions:**
- `draw_header_bar()` - Renders gradient header with title and photo counter
- `draw_footer_bar()` - Displays status bar with dynamic messages and instructions
- `draw_rounded_rectangle()` - Helper function for aesthetic rounded corners
- `display_countdown()` - Shows animated countdown with pulse effect and color changes
- `display_message()` - Animated success message with fade effects

**Core Functions:**
- `initialize_camera()` - Sets up webcam access
- `load_classifiers()` - Loads Haar Cascade models
- `detect_faces_and_smiles()` - Performs detection with stylish visual indicators
- `save_photo()` - Handles photo saving with sequential naming
- `main()` - Orchestrates the entire application flow with UI integration

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
