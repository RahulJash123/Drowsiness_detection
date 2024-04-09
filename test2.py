import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load pre-trained model from JSON and H5 files
def load_model(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    return model

# Load pre-trained eye model
eye_model = load_model("model_a.json", "model_weights.h5")


def detect_eyes(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    if len(faces) == 0:
        cv2.putText(image, '', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        # Detect eyes within face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 0:
            cv2.putText(image, 'Closed.  Drowsiness detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            # Preprocess eye image (resize, normalize, etc.)
            eye_img_gray = cv2.resize(eye_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            eye_img_gray = cv2.cvtColor(eye_img_gray, cv2.COLOR_BGR2GRAY)  # Convert eye image to grayscale
            eye_img_gray = eye_img_gray / 255.0  # Normalize pixel values
            eye_img_gray = np.expand_dims(eye_img_gray, axis=-1)  # Add channel dimension
            eye_img_gray = np.expand_dims(eye_img_gray, axis=0)  # Add batch dimension
            # Predict whether eye is open or closed
            prediction = eye_model.predict(eye_img_gray)
            # Display result
            if prediction[0][0] > 0.5:  
                cv2.putText(image, 'Open', (x + ex, y + ey), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, 'Closed.', (x + ex, y + ey), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
    return image


cap = cv2.VideoCapture(0)


IMAGE_WIDTH = 84
IMAGE_HEIGHT =84

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = detect_eyes(frame)


    cv2.imshow('Eye State Prediction: ', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
