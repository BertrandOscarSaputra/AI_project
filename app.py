from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
from google import genai
from google.genai import types


# Set up Gemini client
client = genai.Client(api_key="AIzaSyAvGclmgjqfF9kT_7m76doTJuNiJ4JB_z0")

app = Flask(__name__)
model = load_model("./Model/inceptionv3_model.h5", compile=False)

def generate_explanation_with_image(img_path, prediction, confidence):
    try:
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        prompt = (
            f"The AI model classified the skin lesion as '{prediction.upper()}' with a confidence of {confidence}%. "
            f"Please confirm and explain whether this classification makes sense based on the uploaded image. "
            f"Describe visible signs or features that align with {prediction} lesions in a clear, medically-informed, readable format. "
            f"Structure your explanation in clear paragraphs and include a disclaimer that this is not a medical diagnosis."
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt
            ]
        )

        return response.text.strip()
    except Exception as e:
        return f"Gemini explanation failed: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    explanation = None
    img_path = None

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
                prediction = class_names[1]
            else:
                prediction = class_names[0]
                confidence = 100 - confidence

            explanation = generate_explanation_with_image(img_path, prediction, confidence)


    # For GET method (i.e., page refresh), all values remain None
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        explanation=explanation,
        img_path=img_path
    )


if __name__ == "__main__":
    app.run(debug=True)
