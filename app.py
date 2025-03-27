import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io

# Initialize Flask App
app = Flask(_name_)

# Load Models
model1 = tf.keras.models.load_model("/content/drive/My Drive/AI_Style_Advisor/Models/face_shape_model.h5", compile=False)
model2 = tf.keras.models.load_model("/content/drive/My Drive/AI_Style_Advisor/Models/hair_attributes_model_fixed.h5", compile=False)
model3 = tf.keras.models.load_model("/content/drive/My Drive/AI_Style_Advisor/Models/skin_tone_classifier_fixed.h5", compile=False)

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
if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000)
