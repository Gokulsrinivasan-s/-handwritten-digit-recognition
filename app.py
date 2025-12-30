from flask import Flask, request, render_template
import joblib
import numpy as np
import cv2 as cv
import os

app = Flask(__name__)

# Load trained ML model
model = joblib.load("model.pkl")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Image preprocessing (same as training)
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (28, 28))
    features = img.flatten().reshape(1, -1) / 255.0

    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction=f"Predicted Digit: {prediction}",
        image_path=file_path
    )

if __name__ == "__main__":
    app.run(debug=True)
