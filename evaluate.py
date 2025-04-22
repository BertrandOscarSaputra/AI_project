from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input  # change if using another model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# ==== CONFIG ====
MODEL_PATH = "./Model/inceptionv3_model.h5"  # or "inceptionv3.h5"
TEST_DIR = "./static/uploads"           # path to your test folder
IMG_SIZE = (224, 224)       # for Xception and InceptionV3
BATCH_SIZE = 32

# ==== LOAD MODEL ====
model = load_model(MODEL_PATH, compile=False)

# ==== LOAD TEST DATA ====
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ==== PREDICT ====
print("Evaluating...")
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# ==== CLASSIFICATION REPORT ====
print("\nClassification Report:")
print(classification_report(test_generator.classes, y_pred, target_names=list(test_generator.class_indices.keys())))

# ==== CONFUSION MATRIX ====
print("Confusion Matrix:")
print(confusion_matrix(test_generator.classes, y_pred))
