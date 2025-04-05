
from flask import Flask, jsonify
from tensorflow import keras
import torch

app = Flask(__name__)

# Load models
audio_model = keras.models.load_model('models/ambulance_siren_model.h5')
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)

@app.route('/')
def index():
    return jsonify({"status": "Backend is running with both models loaded successfully."})

if __name__ == '__main__':
    app.run(debug=True)
