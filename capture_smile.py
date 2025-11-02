"""
Capture Smile AI - Automatic Smile Detection and Photo Capture
A beautiful, modern application with attractive UI/UX for real-time smile detection
"""

import cv2
import os
import numpy as np
from datetime import datetime


def draw_header_bar(frame, photos_captured):
    """
    Draw a modern header bar with app title and photo counter.
    
    Args:
        frame: The video frame to draw on
        photos_captured: Number of photos captured so far
    """
    height, width = frame.shape[:2]
    
    # Create gradient background for header (dark blue to lighter blue)
    header_height = 80
    overlay = frame.copy()
    
    # Draw gradient header
    for i in range(header_height):
        alpha = i / header_height
        color_val = int(40 + (20 * alpha))
        cv2.rectangle(overlay, (0, i), (width, i + 1), (color_val, color_val, 60), -1)
    
    # Blend overlay with frame
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Draw app title
    title = "CAPTURE SMILE AI"
   cv2.putText(frame, title, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, title, (20, 45), cv2.FONT_HERSHEY_BOLD, 1.2, (100, 200, 255), 2)
    
    # Draw photo counter on the right
    counter_text = f"Photos: {photos_captured}"
    text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    counter_x = width - text_size[0] - 20
    cv2.putText(frame, counter_text, (counter_x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def draw_footer_bar(frame, status_text):
    """
    Draw a modern footer bar with status and instructions.
    
    Args:
        frame: The video frame to draw on
        status_text: Current status message to display
    """
    height, width = frame.shape[:2]
    footer_height = 70
    footer_y = height - footer_height
    
    # Create gradient background for footer
    overlay = frame.copy()
    for i in range(footer_height):
        alpha = i / footer_height
        color_val = int(60 - (20 * alpha))
        cv2.rectangle(overlay, (0, footer_y + i), (width, footer_y + i + 1), (color_val, color_val, 40), -1)
    
    # Blend overlay with frame
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Draw status text (centered)
    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    status_x = (width - status_size[0]) // 2
    cv2.putText(frame, status_text, (status_x, footer_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)
    
    # Draw instruction text at bottom
    instruction = "Press 'Q' to Quit  |  Smile to Capture"
    inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    inst_x = (width - inst_size[0]) // 2
    cv2.putText(frame, instruction, (inst_x, footer_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_rounded_rectangle(frame, x, y, w, h, color, thickness=2, corner_radius=15):
    """
    Draw a rectangle with rounded corners for better aesthetics.
    
    Args:
        frame: The video frame to draw on
        x, y, w, h: Rectangle coordinates and dimensions
        color: Color tuple (B, G, R)
        thickness: Line thickness
        corner_radius: Radius of rounded corners
    """
    # Draw the four sides
    cv2.line(frame, (x + corner_radius, y), (x + w - corner_radius, y), color, thickness)
    cv2.line(frame, (x + corner_radius, y + h), (x + w - corner_radius, y + h), color, thickness)
    cv2.line(frame, (x, y + corner_radius), (x, y + h - corner_radius), color, thickness)
    cv2.line(frame, (x + w, y + corner_radius), (x + w, y + h - corner_radius), color, thickness)
    
    # Draw the four corners
    cv2.ellipse(frame, (x + corner_radius, y + corner_radius), (corner_radius, corner_radius), 180, 0, 90, color, thickness)
    cv2.ellipse(frame, (x + w - corner_radius, y + corner_radius), (corner_radius, corner_radius), 270, 0, 90, color, thickness)
    cv2.ellipse(frame, (x + corner_radius, y + h - corner_radius), (corner_radius, corner_radius), 90, 0, 90, color, thickness)
    cv2.ellipse(frame, (x + w - corner_radius, y + h - corner_radius), (corner_radius, corner_radius), 0, 0, 90, color, thickness)


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
    Detect faces and smiles in the given frame with beautiful visual indicators.
    
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
        # Draw a modern rounded rectangle around the face with glow effect
        # Outer glow (cyan)
        draw_rounded_rectangle(frame, x-2, y-2, w+4, h+4, (255, 200, 100), thickness=3, corner_radius=20)
        # Inner border (bright cyan)
        draw_rounded_rectangle(frame, x, y, w, h, (255, 255, 100), thickness=2, corner_radius=18)
        
        # Add "FACE DETECTED" label above the face
        label = "FACE DETECTED"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_x = x + (w - label_size[0]) // 2
        label_y = y - 10
        if label_y > 20:
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
        
        # Extract the region of interest (ROI) - the face area
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        
        # Detect smiles within the face region
        # Using stricter parameters for more accurate smile detection
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        
        # Loop through detected smiles
        for (sx, sy, sw, sh) in smiles:
            # Draw a stylish rectangle around the smile with double border
            # Outer glow (pink)
            cv2.rectangle(roi_color, (sx-2, sy-2), (sx + sw+2, sy + sh+2), (200, 100, 255), 3)
            # Inner border (bright pink)
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 150, 255), 2)
            
            # Add smile icon indicator
            smile_icon = "^_^"
            icon_size = cv2.getTextSize(smile_icon, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            icon_x = sx + (sw - icon_size[0]) // 2
            icon_y = sy + sh + 20
            if icon_y < h - 5:
                cv2.putText(roi_color, smile_icon, (icon_x, icon_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(roi_color, smile_icon, (icon_x, icon_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 1)
            
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


def display_countdown(frame, countdown_value, countdown_frames):
    """
    Display an attractive animated countdown number on the frame (3, 2, 1).
    
    Args:
        frame: The video frame to draw on
        countdown_value: The countdown number to display (3, 2, or 1)
        countdown_frames: Current frame count for pulse animation
    """
    if countdown_value > 0:
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Create pulse effect (size changes based on frame count)
        pulse = 1.0 + (0.3 * abs((countdown_frames % 20) - 10) / 10)
        
        # Set text properties for large countdown with pulse effect
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = 8.0 * pulse  # Very large font with animation
        thickness = int(15 * pulse)
        
        # Color changes based on countdown value
        colors = {
            3: (100, 255, 100),   # Green for 3
            2: (100, 200, 255),   # Yellow for 2
            1: (100, 100, 255)    # Red for 1
        }
        color = colors.get(countdown_value, (100, 255, 100))
        
        # Get text size to center it
        text = str(countdown_value)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # Draw glowing circle background
        circle_radius = int(150 * pulse)
        cv2.circle(frame, (width // 2, height // 2), circle_radius + 20, (50, 50, 50), -1)
        cv2.circle(frame, (width // 2, height // 2), circle_radius + 15, color, 5)
        cv2.circle(frame, (width // 2, height // 2), circle_radius + 10, (255, 255, 255), 2)
        
        # Draw countdown with shadow and glow effect
        # Shadow
        cv2.putText(frame, text, (text_x + 5, text_y + 5), font, font_scale, (0, 0, 0), thickness + 8)
        # Outer glow
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 4)
        # Main number
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Add "Get Ready!" text below
        ready_text = "GET READY!"
        ready_size = cv2.getTextSize(ready_text, cv2.FONT_HERSHEY_BOLD, 1.2, 2)[0]
        ready_x = (width - ready_size[0]) // 2
        ready_y = text_y + 100
        cv2.putText(frame, ready_text, (ready_x, ready_y), cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 3)
        cv2.putText(frame, ready_text, (ready_x, ready_y), cv2.FONT_HERSHEY_BOLD, 1.2, (100, 255, 255), 2)


def display_message(frame, message, duration_counter, max_duration=30):
    """
    Display an attractive success message on the frame for a certain duration.
    
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
        
        # Create fade-in fade-out effect
        if duration_counter > max_duration - 10:
            alpha = (max_duration - duration_counter) / 10
        elif duration_counter < 10:
            alpha = duration_counter / 10
        else:
            alpha = 1.0
        
        # Set text properties
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = 2.0
        thickness = 4
        
        # Get text size to center it
        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 150
        
        # Draw background box with transparency
        padding = 20
        box_x1 = text_x - padding
        box_y1 = text_y - text_size[1] - padding
        box_x2 = text_x + text_size[0] + padding
        box_y2 = text_y + padding
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (50, 200, 50), -1)
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (100, 255, 100), 3)
        cv2.addWeighted(overlay, 0.7 * alpha, frame, 1 - (0.7 * alpha), 0, frame)
        
        # Draw checkmark icon
        checkmark = "✓"
        check_size = cv2.getTextSize(checkmark, font, 2.0, 4)[0]
        check_x = text_x - check_size[0] - 30
        cv2.putText(frame, checkmark, (check_x, text_y), font, 2.0, (100, 255, 100), 6)
        
        # Draw text with shadow and glow effect
        cv2.putText(frame, message, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, message, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, message, (text_x, text_y), font, font_scale, (100, 255, 100), thickness - 1)
        
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
    print("- Smile at the camera to trigger a countdown!")
    print("- A 3-2-1 countdown will appear before capturing")
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
    countdown_timer = 0  # Countdown value (3, 2, 1, or 0 when not counting)
    countdown_frames = 0  # Frame counter for countdown timing
    capture_next_frame = False  # Flag to capture photo on next frame (after countdown)
    
    print("Starting live camera feed... Press 'q' to quit.\n")
    
    # Main loop - continuously capture and process frames
    while True:
        # Read a frame from the camera
        success, frame = camera.read()
        
        # Check if frame was read successfully
        if not success:
            print("Error: Failed to read frame from camera!")
            break
        
        # If we need to capture a photo this frame (after countdown completed)
        if capture_next_frame:
            # Capture the photo (frame is clean, no countdown overlay)
            photo_counter = save_photo(frame, photo_counter)
            
            # Set message display duration (30 frames ≈ 1 second at 30fps)
            message_duration = 30
            
            # Set cooldown to prevent multiple captures (90 frames ≈ 3 seconds)
            smile_cooldown = 90
            
            # Reset the capture flag
            capture_next_frame = False
        
        # Detect faces and smiles in the current frame
        faces, smile_detected = detect_faces_and_smiles(frame, face_cascade, smile_cascade)
        
        # If a smile is detected and cooldown has expired and no countdown is active
        if smile_detected and smile_cooldown == 0 and countdown_timer == 0:
            # Start the countdown at 3
            countdown_timer = 3
            countdown_frames = 0
            print("Smile detected! Starting countdown...")
        
        # Handle countdown logic
        if countdown_timer > 0:
            # Display the current countdown number with animation
            display_countdown(frame, countdown_timer, countdown_frames)
            
            # Increment frame counter
            countdown_frames += 1
            
            # Change countdown number every 30 frames (≈ 1 second at 30fps)
            if countdown_frames >= 30:
                countdown_timer -= 1
                countdown_frames = 0
                
                # If countdown finished, set flag to capture on NEXT frame
                if countdown_timer == 0:
                    capture_next_frame = True
        
        # Decrease cooldown counter
        if smile_cooldown > 0:
            smile_cooldown -= 1
        
        # Display "Photo Captured!" message if active (only when not counting down)
        if countdown_timer == 0:
            message_duration = display_message(frame, "Photo Captured!", message_duration)
        
        # Determine current status for footer
        if countdown_timer > 0:
            status_text = f"📸 COUNTDOWN: {countdown_timer}"
        elif len(faces) > 0:
            if smile_detected:
                status_text = "😊 SMILE DETECTED - Keep Smiling!"
            else:
                status_text = "😐 Face Detected - SMILE to Capture!"
        else:
            status_text = "👤 Looking for Faces..."
        
        # Draw the beautiful UI elements
        draw_header_bar(frame, photo_counter - 1)
        draw_footer_bar(frame, status_text)
        
        # Display the frame in a window with custom name
        cv2.imshow('Capture Smile AI - Professional Edition', frame)
        
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
