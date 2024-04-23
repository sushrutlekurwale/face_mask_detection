import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained mask detection model
model = load_model('mask_detection_model.h5')

# Function to detect face masks
def detect_mask(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Initialize variables to track mask detection result
    mask_detected = False
    
    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_roi = frame[y:y+h, x:x+w]
        
        # Resize the face region to match model input shape
        resized_face = cv2.resize(face_roi, (100, 100))
        
        # Preprocess the resized face
        resized_face = resized_face / 255.0
        reshaped_face = np.expand_dims(resized_face, axis=0)
        
        # Perform mask detection on the face region
        result = model.predict(reshaped_face)
        
        # Process the result
        label = "Mask" if result[0][0] > 0.5 else "NO Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # Set mask_detected flag if mask is detected
        if label == "Mask":
            mask_detected = True
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Display the mask detection result
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    # Print message indicating mask detection result
    if mask_detected:
        print("Move On Entry Accepted")
    else:
        print("Stay Back Entry Declined")
    
    return frame

# Initialize the video stream
vs = cv2.VideoCapture(0)

while True:
    # Read a single frame from the video stream
    ret, frame = vs.read()
    
    # Check if the frame is successfully captured
    if ret:
        # Detect face masks in the captured frame
        frame = detect_mask(frame)
        
        # Display the frame with mask detection results
        cv2.imshow('Face Mask Detection', frame)
    
    else:
        print("Failed to capture frame from the webcam.")
    
    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream
vs.release()
cv2.destroyAllWindows()
