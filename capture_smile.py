"""
Capture Smile AI - Automatic Smile Detection and Photo Capture
A beginner-friendly Python application using OpenCV for real-time smile detection
"""

import cv2
import os
from datetime import datetime


def initialize_camera():
    """
    Initialize and configure the webcam.
    Returns the camera object for video capture.
    """
    # Create a video capture object (0 is the default camera)
    camera = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error: Could not access the webcam!")
        return None
    
    print("Camera initialized successfully!")
    return camera


def load_classifiers():
    """
    Load the Haar Cascade classifiers for face and smile detection.
    Returns face_cascade and smile_cascade objects.
    """
    # Load pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load pre-trained Haar Cascade classifier for smile detection
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    # Verify that classifiers loaded successfully
    if face_cascade.empty() or smile_cascade.empty():
        print("Error: Could not load Haar Cascade classifiers!")
        return None, None
    
    print("Haar Cascade classifiers loaded successfully!")
    return face_cascade, smile_cascade


def detect_faces_and_smiles(frame, face_cascade, smile_cascade):
    """
    Detect faces and smiles in the given frame.
    
    Args:
        frame: The video frame to analyze
        face_cascade: Haar Cascade classifier for faces
        smile_cascade: Haar Cascade classifier for smiles
    
    Returns:
        faces: List of detected face rectangles
        smile_detected: Boolean indicating if a smile was found
    """
    # Convert frame to grayscale (Haar Cascades work better with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    # Parameters: scaleFactor=1.3 (how much image is reduced at each scale)
    #            minNeighbors=5 (how many neighbors each candidate rectangle should have)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    smile_detected = False
    
    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a green rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract the region of interest (ROI) - the face area
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        # Detect smiles within the face region
        # Using stricter parameters for more accurate smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        
        # Loop through detected smiles
        for (sx, sy, sw, sh) in smiles:
            # Draw a blue rectangle around the smile
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
            smile_detected = True
    
    return faces, smile_detected


def save_photo(frame, photo_counter):
    """
    Save the captured photo to disk with a sequential filename.
    
    Args:
        frame: The video frame to save
        photo_counter: Current count of saved photos
    
    Returns:
        Updated photo_counter
    """
    # Create a 'captured_smiles' folder if it doesn't exist
    if not os.path.exists('captured_smiles'):
        os.makedirs('captured_smiles')
    
    # Generate filename with counter (smile_1.jpg, smile_2.jpg, etc.)
    filename = f'captured_smiles/smile_{photo_counter}.jpg'
    
    # Save the image
    cv2.imwrite(filename, frame)
    
    print(f"Photo saved: {filename}")
    
    return photo_counter + 1


def display_message(frame, message, duration_counter, max_duration=30):
    """
    Display a message on the frame for a certain duration.
    
    Args:
        frame: The video frame to draw on
        message: Text message to display
        duration_counter: Current frame count for the message
        max_duration: How many frames to show the message
    
    Returns:
        Updated duration_counter
    """
    if duration_counter > 0:
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Set text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        color = (0, 255, 255)  # Yellow color
        thickness = 3
        
        # Get text size to center it
        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 50
        
        # Draw text with black outline for better visibility
        cv2.putText(frame, message, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, message, (text_x, text_y), font, font_scale, color, thickness)
        
        return duration_counter - 1
    
    return 0


def main():
    """
    Main function to run the Capture Smile AI application.
    """
    print("=" * 60)
    print("      Welcome to Capture Smile AI!")
    print("=" * 60)
    print("\nInstructions:")
    print("- The camera will start automatically")
    print("- Smile at the camera to capture a photo!")
    print("- Press 'q' to quit the application")
    print("- Photos will be saved in the 'captured_smiles' folder")
    print("=" * 60)
    print()
    
    # Initialize the camera
    camera = initialize_camera()
    if camera is None:
        return
    
    # Load Haar Cascade classifiers
    face_cascade, smile_cascade = load_classifiers()
    if face_cascade is None or smile_cascade is None:
        camera.release()
        return
    
    # Initialize counters
    photo_counter = 1
    message_duration = 0
    smile_cooldown = 0  # Cooldown to prevent multiple captures of the same smile
    
    print("Starting live camera feed... Press 'q' to quit.\n")
    
    # Main loop - continuously capture and process frames
    while True:
        # Read a frame from the camera
        success, frame = camera.read()
        
        # Check if frame was read successfully
        if not success:
            print("Error: Failed to read frame from camera!")
            break
        
        # Detect faces and smiles in the current frame
        faces, smile_detected = detect_faces_and_smiles(frame, face_cascade, smile_cascade)
        
        # If a smile is detected and cooldown has expired
        if smile_detected and smile_cooldown == 0:
            # Save the photo
            photo_counter = save_photo(frame, photo_counter)
            
            # Set message display duration (30 frames ≈ 1 second at 30fps)
            message_duration = 30
            
            # Set cooldown to prevent multiple captures (60 frames ≈ 2 seconds)
            smile_cooldown = 60
        
        # Decrease cooldown counter
        if smile_cooldown > 0:
            smile_cooldown -= 1
        
        # Display "Smile detected!" message if active
        message_duration = display_message(frame, "Smile detected!", message_duration)
        
        # Add instructions text at the bottom of the frame
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the frame in a window
        cv2.imshow('Capture Smile AI - Live Feed', frame)
        
        # Wait for 1ms and check if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nQuitting application...")
            break
    
    # Clean up resources
    print("Releasing camera and closing windows...")
    camera.release()
    cv2.destroyAllWindows()
    
    print(f"\nTotal photos captured: {photo_counter - 1}")
    print("Thank you for using Capture Smile AI!")


# Entry point of the program
if __name__ == "__main__":
    main()
