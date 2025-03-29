from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import gdown
import os
import requests  # ✅ Needed for downloading image from URL

app = Flask(__name__)

# Ensure model folder
os.makedirs("Models", exist_ok=True)

# Download face shape model from Google Drive if not present
model_path = "Models/face_shape_model.h5"
if not os.path.exists(model_path):
    gdown.download(
        "https://drive.google.com/uc?id=1yJOmb0sz2PpQKaMHQzwv3bi-MET6dVX1",
        model_path,
        quiet=False
    )

# Load models
face_shape_model = tf.keras.models.load_model(model_path)
skin_tone_model = tf.keras.models.load_model("Models/skin_tone_classifier.h5")
hair_model = tf.keras.models.load_model("Models/hair_attributes_model.h5")

# Class labels
face_shape_classes = ["Round", "Oval", "Square", "Heart", "Oblong"]
skin_tone_classes = ["Black", "White", "Brown"]
hair_classes = [
    "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair",
    "Gray_Hair", "Straight_Hair", "Wavy_Hair", "Receding_Hairline"
]

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Prediction function
def run_prediction(image):
    face_pred = face_shape_model.predict(image)
    skin_pred = skin_tone_model.predict(image)
    hair_pred = hair_model.predict(image)

    return {
        "Face Shape": face_shape_classes[np.argmax(face_pred)],
        "Skin Tone": skin_tone_classes[np.argmax(skin_pred)],
        "Hair Features": [
            hair_classes[i] for i, val in enumerate(hair_pred[0]) if val > 0.5
        ]
    }

# ✅ Route 1: For base64 input (your original code)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        base64_str = data.get("image_base64")

        if not base64_str:
            return jsonify({"error": "No image_base64 received"})

        base64_data = base64_str.split(",")[-1]
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes))
        image = preprocess_image(image)

        result = run_prediction(image)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ Route 2: For Wix — use image URL instead of base64
@app.route("/predict-from-url", methods=["POST"])
def predict_from_url():
    try:
        data = request.get_json()
        image_url = data.get("image_url")

        if not image_url:
            return jsonify({"error": "No image_url provided"}), 400

        # Download image from URL
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        image = preprocess_image(image)

        result = run_prediction(image)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run locally (not used on Render)
if __name__ == "__main__":
    app.run()
