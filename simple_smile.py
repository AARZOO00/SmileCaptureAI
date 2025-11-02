import cv2
import numpy as np
import time
import os

# Create folder for saving photos
if not os.path.exists('captured_smiles'):
    os.makedirs('captured_smiles')

print("🎉 SmileCaptureAI Started!")
print("📸 Smile at your camera!")
print("⏹️  Press 'q' to quit")

# Load classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
photo_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam access failed!")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20, minSize=(25, 15))
        
        if len(smiles) > 0:
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
            
            # Countdown
            for i in range(3, 0, -1):
                temp_frame = frame.copy()
                cv2.putText(temp_frame, str(i), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                cv2.imshow('SmileCaptureAI', temp_frame)
                cv2.waitKey(1000)
            
            # Save photo
            filename = f"captured_smiles/smile_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            photo_count += 1
            print(f"✅ Photo {photo_count} saved!")
    
    cv2.imshow('SmileCaptureAI', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal photos captured: {photo_count}")
print("Thank you for using SmileCaptureAI!")