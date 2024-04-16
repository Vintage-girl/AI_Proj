# import cv2
# import numpy as np
# from tensorflow import keras


# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# new_model = keras.models.load_model("my_model.h5")
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     raise IOError("Cannot open webcam")

# while True:
#     ret, frame = cap.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = faceCascade.detectMultiScale(gray, 1.1, 4)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
        
#         eyes = eyeCascade.detectMultiScale(roi_gray)

#         for (ex, ey, ew, eh) in eyes:
#             # cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
#             eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            
#             final_image = cv2.resize(eye_roi, (224, 224))
#             final_image = np.expand_dims(final_image, axis=0)  
#             final_image = final_image / 255.0  
            
#             predictions = new_model.predict(final_image)
            
#             status = "Sober" if predictions > -3 else "Intoxicated"
            
#             cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
#     cv2.imshow('Face Cam', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from tensorflow import keras

# Load the pre-trained model
new_model = keras.models.load_model("my_model.h5")

# Load the Haar cascades for face and eye detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Determine the color based on intoxication status
        color = (0, 255, 0)  # Default color (green)
        
        # Extract the region of interest (ROI) for eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes in the ROI
        eyes = eyeCascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            
            # Resize and preprocess the eye image for prediction
            final_image = cv2.resize(eye_roi, (224, 224))
            final_image = np.expand_dims(final_image, axis=0)  
            final_image = final_image / 255.0  
            
            # Predict intoxication status
            predictions = new_model.predict(final_image)
            
            # Determine status text and color based on predictions
            status = "Sober" if predictions > -3 else "Intoxicated"
            
            if status == "Intoxicated":
                color = (0, 0, 255)  # Change color to red
                
            # Draw rectangle around the face with determined color
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw the status text on the frame
            cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Display the frame with annotations
    cv2.imshow('Face Cam', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

