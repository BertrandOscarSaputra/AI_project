from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
from google import genai

# Set up Gemini client
client = genai.Client(api_key="AIzaSyAvGclmgjqfF9kT_7m76doTJuNiJ4JB_z0")

app = Flask(__name__)
model = load_model("./Model/inceptionv3_model.h5", compile=False)

def generate_explanation(prediction, confidence):
    try:
        prompt = (
            f"The AI model classified the skin lesion as '{prediction.upper()}' with a confidence of {confidence}%. "
            f"Explain clearly and concisely why the lesion is likely {prediction}. "
            f"Break down the explanation into clear, readable paragraphs. "
            f"Include common characteristics of {prediction} skin lesions. "
            f"Format the response with clear sections or bullet points for easy reading, like a medical summary."
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text.strip()
    except Exception as e:
        return f"Explanation generation failed: {e}"


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

            explanation = generate_explanation(prediction, confidence)

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
