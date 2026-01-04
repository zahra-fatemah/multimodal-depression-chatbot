import librosa
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/audio_emotion_model.h5")

emotion_to_risk = {
    "neutral": 0,
    "calm": 0,
    "happy": 0,
    "sad": 2,
    "fear": 2,
    "angry": 1
}

emotion_labels = list(emotion_to_risk.keys())

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def audio_risk_score(file_path):
    mfcc = extract_mfcc(file_path).reshape(1, -1)
    prediction = model.predict(mfcc)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    return emotion_to_risk[emotion]
