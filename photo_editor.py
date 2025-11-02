import cv2
import numpy as np
import os

class PhotoEditor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.cap = cv2.VideoCapture(0)
        self.current_filter = "normal"
        self.filters = ["normal", "grayscale", "sepia", "warm", "cool", "vintage", "blur"]
    
    def apply_filter(self, frame, filter_name):
        if filter_name == "normal":
            return frame
        elif filter_name == "grayscale":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif filter_name == "sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            return cv2.transform(frame, kernel)
        elif filter_name == "warm":
            frame[:, :, 0] = cv2.multiply(frame[:, :, 0], 0.9)  # Reduce blue
            frame[:, :, 2] = cv2.multiply(frame[:, :, 2], 1.1)  # Increase red
            return frame
        elif filter_name == "cool":
            frame[:, :, 0] = cv2.multiply(frame[:, :, 0], 1.1)  # Increase blue
            frame[:, :, 2] = cv2.multiply(frame[:, :, 2], 0.9)  # Reduce red
            return frame
        elif filter_name == "vintage":
            # Add vintage effect
            frame = self.apply_sepia(frame)
            noise = np.random.randint(0, 50, frame.shape, dtype='uint8')
            return cv2.add(frame, noise)
        elif filter_name == "blur":
            return cv2.GaussianBlur(frame, (15, 15), 0)
    
    def add_sticker(self, frame, faces):
        for (x, y, w, h) in faces:
            # Add sunglasses sticker
            sticker = cv2.imread('sunglasses.png', cv2.IMREAD_UNCHANGED)
            if sticker is not None:
                sticker = cv2.resize(sticker, (w, h//3))
                # Overlay sticker on face
                overlay(frame, sticker, x, y + h//6)
        return frame
    
    def overlay(self, background, overlay, x, y):
        # Simple overlay function
        bg_height, bg_width = background.shape[:2]
        if x >= bg_width or y >= bg_height:
            return
        
        h, w = overlay.shape[:2]
        if x + w > bg_width:
            w = bg_width - x
            overlay = overlay[:, :w]
        if y + h > bg_height:
            h = bg_height - y
            overlay = overlay[:h]
        
        if overlay.shape[2] == 4:  # With alpha channel
            alpha = overlay[:, :, 3] / 255.0
            for c in range(3):
                background[y:y+h, x:x+w, c] = (alpha * overlay[:, :, c] + 
                                              (1 - alpha) * background[y:y+h, x:x+w, c])
        else:
            background[y:y+h, x:x+w] = overlay
    
    def run(self):
        print("🎨 Photo Editor with Filters!")
        print("Filters: 1-Normal 2-Grayscale 3-Sepia 4-Warm 5-Cool 6-Vintage 7-Blur")
        print("Press 'c' to capture, 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Apply current filter
            filtered_frame = self.apply_filter(frame.copy(), self.current_filter)
            
            # Face detection on original frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Draw face rectangles on filtered frame
            for (x, y, w, h) in faces:
                cv2.rectangle(filtered_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Smile detection
                roi_gray = gray[y:y+h, x:x+w]
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                if len(smiles) > 0:
                    cv2.putText(filtered_frame, "SMILE! 😊", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display current filter name
            cv2.putText(filtered_frame, f"Filter: {self.current_filter.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Photo Editor', filtered_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture photo
                filename = f"edited_photo_{int(time.time())}.jpg"
                cv2.imwrite(filename, filtered_frame)
                print(f"✅ Photo saved: {filename}")
            elif key in [ord(str(i)) for i in range(1, 8)]:
                # Change filter
                filter_index = key - ord('1')
                if filter_index < len(self.filters):
                    self.current_filter = self.filters[filter_index]
                    print(f"🔧 Filter changed to: {self.current_filter}")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    editor = PhotoEditor()
    editor.run()