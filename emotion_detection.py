import cv2
import numpy as np
import time
import os
from deepface import DeepFace

# Install: pip install deepface

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'surprise': (255, 255, 0), # Cyan
            'neutral': (255, 255, 255) # White
        }
    
    def detect_emotion(self, frame):
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']
            return dominant_emotion
        except:
            return 'neutral'
    
    def run(self):
        print("🎭 Emotion Detection Started!")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_frame = frame[y:y+h, x:x+w]
                emotion = self.detect_emotion(face_frame)
                
                # Draw rectangle with emotion color
                color = self.emotion_colors.get(emotion, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Display emotion text
                cv2.putText(frame, f"Emotion: {emotion.upper()}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Display emoji based on emotion
                emojis = {
                    'happy': '😊',
                    'sad': '😢', 
                    'angry': '😠',
                    'surprise': '😮',
                    'neutral': '😐'
                }
                emoji = emojis.get(emotion, '😐')
                cv2.putText(frame, emoji, (x+w-30, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Emotion Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.run()