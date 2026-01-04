import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load face emotion model
model = load_model("models/facial_emotion_cnn_improved.h5")

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(
    r"C:\Users\VICTUS\Downloads\multimodal_depression_chatbot\haarcascade_frontalface_default.xml"
)

# Emotion index → depression risk
emotion_to_risk = {
    0: 1,  # angry
    1: 1,  # disgust
    2: 2,  # fear
    3: 0,  # happy
    4: 2,  # sad
    5: 1,  # surprise
    6: 0   # neutral
}

def video_risk_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    if len(faces) == 0:
        return None  # No face detected

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face.reshape(1, 48, 48, 1) / 255.0

    prediction = model.predict(face, verbose=0)
    emotion = np.argmax(prediction)

    return emotion_to_risk[emotion]
