from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load model
model = load_model('./Model/inceptionv3_model.h5')

# Replace with your actual class names
class_names = ['benign', 'malignant']

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction = model_predict(filepath, model)
            return render_template('index.html', prediction=prediction, image_url=filepath)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
