import cv2

# face detection model
face_detector=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up the webcam
cap = cv2.VideoCapture(0)

# Start streaming the webcam
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame = cv2.flip(frame, 1)
    
    # convert frame into gray for face detection purposes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # draw rectangle on detected  faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  
    
    cv2.imshow('Frame', frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
