import cv2
import numpy as np

print("🎭 Starting Emotion Detection Debug...")

# Load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

print("✅ Classifiers loaded")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not captured")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    print(f"Faces detected: {len(faces)}")
    
    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Detect smiles in face ROI
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20, minSize=(25, 15))
        
        print(f"Smiles detected in face: {len(smiles)}")
        
        if len(smiles) > 0:
            emotion = "HAPPY"
            emoji = "😊"
            color = (0, 255, 0)  # Green
        else:
            emotion = "NEUTRAL" 
            emoji = "😐"
            color = (255, 255, 255)  # White
        
        # Draw emotion info - LARGE TEXT
        cv2.putText(frame, f"{emotion} {emoji}", (x, y-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)  # Increased size
    
    # Display frame
    cv2.imshow('Emotion Detection DEBUG - Press Q to quit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Debug completed")