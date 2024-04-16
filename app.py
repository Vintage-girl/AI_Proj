from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Load the pre-trained model
new_model = keras.models.load_model("my_model.h5")

# Load the Haar cascades for face and eye detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Placeholder for camera object and status
camera = None
is_camera_on = False

def generate_frames():
    global camera
    while is_camera_on:
        ret, frame = camera.read()
        if not ret:
            break

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

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Redirect to home page after login
        return redirect(url_for('home'))
    else:
        return render_template('login.html')

@app.route('/home')
def home():
    # Render the home page
    return render_template('home.html')

@app.route('/signup')
def signup():
    # Render the sign up page
    return render_template('signup.html')

@app.route('/index')
def index():
    # Render the index page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control_camera():
    global camera, is_camera_on
    if request.form.get('action') == 'start':
        camera = cv2.VideoCapture(0)
        is_camera_on = True
    elif request.form.get('action') == 'stop':
        if camera is not None:
            camera.release()
        is_camera_on = False
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
