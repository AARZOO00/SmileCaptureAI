import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("✅ Camera connected!")
    
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera Test - PRESS Q TO CLOSE', frame)
        
        # Q press karo to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("❌ Camera not working!")