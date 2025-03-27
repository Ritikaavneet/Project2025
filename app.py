import os
import gdown
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io

# Initialize Flask App
app = Flask(__name__)

# Google Drive File ID for face shape model
FACE_SHAPE_MODEL_ID = "1yJOmb0sz2PpQKaMHQzwv3bi-MET6dVX1" 
MODEL_PATH_1 = "face_shape_model.h5"
MODEL_PATH_2 = "hair_attributes_model.h5"
MODEL_PATH_3 = "skin_tone_classifier.h5"

# Download face shape model if it doesn't exist
if not os.path.exists(MODEL_PATH_1):
    print("Downloading face shape model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FACE_SHAPE_MODEL_ID}", MODEL_PATH_1, quiet=False)

# Load Models
model1 = tf.keras.models.load_model(MODEL_PATH_1, compile=False)
model2 = tf.keras.models.load_model(MODEL_PATH_2, compile=False)
model3 = tf.keras.models.load_model(MODEL_PATH_3, compile=False)

# Recompile Models
model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Prediction Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read())).resize((224, 224))  # Resize to model's input size
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Run Predictions
        face_pred = model1.predict(img_array)
        hair_pred = model2.predict(img_array)
        skin_pred = model3.predict(img_array)

        # Convert to Human-Readable Output (modify labels as needed)
        face_shape = ["Round", "Oval", "Square", "Heart", "Diamond"][np.argmax(face_pred)]
        hair_attribute = ["Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair", "Straight_Hair", "Wavy_Hair", "Receding_Hairline"][np.argmax(hair_pred)]
        skin_tone = ["black", "brown", "white"][np.argmax(skin_pred)]

        return jsonify({
            "Face Shape": face_shape,
            "Hair Attribute": hair_attribute,
            "Skin Tone": skin_tone
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    
# Run Flask App
if __name__ == "__main__":  
    app.run(host="0.0.0.0", port=5000, debug=True)
