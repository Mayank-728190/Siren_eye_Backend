from flask import Flask, request, jsonify
import torch
from ultralytics import YOLO
import numpy as np
import librosa
from keras.models import load_model
import tempfile
import os
import moviepy.editor as mp

app = Flask(__name__)

# Load models once
yolo_model = YOLO("models/best.pt")
audio_model = load_model("models/ambulance_siren_model.h5")

SAMPLE_RATE = 22050

@app.route("/detect", methods=["POST"])
def detect():
    video = request.files.get("video")
    if not video:
        return jsonify({"error": "No video uploaded"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        video.save(temp_video.name)
        video_path = temp_video.name

    try:
        result = run_detection(video_path)
        os.remove(video_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.predict(frame, conf=0.65)
        for result in results:
            if len(result.boxes) > 0:
                detected = True
                break

    cap.release()

    is_siren = classify_audio(video_path) if detected else False
    return {"ambulance_detected": detected, "siren_detected": is_siren}

def classify_audio(video_path):
    video = mp.VideoFileClip(video_path)
    temp_audio_path = video_path.replace(".mp4", "_temp.wav")
    video.audio.write_audiofile(temp_audio_path, fps=SAMPLE_RATE)

    y, sr = librosa.load(temp_audio_path, sr=SAMPLE_RATE)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 128 - log_mel_spec.shape[1])), mode='constant')
    mel_spec_reshaped = np.expand_dims(log_mel_spec, axis=-1)
    prediction = audio_model.predict(np.expand_dims(mel_spec_reshaped, axis=0))

    os.remove(temp_audio_path)
    return prediction[0][0] > 0.5

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
