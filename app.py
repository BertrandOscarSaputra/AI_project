from flask import Flask, render_template, request
from gpt_helper import get_explanation
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os


app = Flask(__name__)
model = load_model("./Model/inceptionv3_model.h5", compile=False)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None
    explanation = None

    if request.method == "POST":
        img_file = request.files["image"]
        if img_file:
            img_path = os.path.join("static/uploads", img_file.filename)
            img_file.save(img_path)

            img = image.load_img(img_path, target_size=(224, 224))  
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)  

            preds = model.predict(img_array)

            class_names = ["benign", "malignant"]
            confidence = round(float(preds[0][0]) * 100, 2)

            if preds[0][0] > 0.5:
                prediction = class_names[1]  # malignant
            else:
                prediction = class_names[0]  # benign
                confidence = 100 - confidence

            # Ambil penjelasan dari GPT
            explanation = get_explanation(prediction)

    return render_template("index.html", prediction=prediction, confidence=confidence, img_path=img_path, explanation=explanation)

if __name__ == "__main__":
    app.run(debug=True)