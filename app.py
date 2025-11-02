from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Create folders
if not os.path.exists('static/captured_smiles'):
    os.makedirs('static/captured_smiles')

class SmileDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.cap = cv2.VideoCapture(0)
        self.photo_count = 0
    
    def generate_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            
            # Face and smile detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20, minSize=(25, 15))
                if len(smiles) > 0:
                    cv2.putText(frame, "SMILE DETECTED!", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

detector = SmileDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detector.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture_photo():
    try:
        success, frame = detector.cap.read()
        if success:
            filename = f"smile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = f"static/captured_smiles/{filename}"
            cv2.imwrite(filepath, frame)
            detector.photo_count += 1
            return jsonify({
                'success': True, 
                'filename': f"static/captured_smiles/{filename}", 
                'count': detector.photo_count
            })
        return jsonify({'success': False, 'error': 'Camera error'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)