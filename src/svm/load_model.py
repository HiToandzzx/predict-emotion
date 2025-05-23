from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image

app = Flask(__name__)

# Load mô hình
model_path = "svm_dog_emotion_model.pkl"
model = joblib.load(model_path)


# Tiền xử lý ảnh
def preprocess_image(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    return gray


# Trích xuất đặc trưng HOG
def extract_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features


# Trang chủ API
@app.route("/")
def home():
    return "Welcome to Dog Emotion Prediction API"


# API test method GET
@app.route("/predict", methods=["GET"])
def test_predict():
    return "Please send a POST request with an image file."


# API dự đoán cảm xúc
@app.route("/predict", methods=["POST"])
def predict_emotion():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file.stream)
    image = np.array(image)

    # Kiểm tra nếu ảnh có alpha channel, chuyển về RGB
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Tiền xử lý ảnh
    img_processed = preprocess_image(image)
    features = extract_features(img_processed)

    # Dự đoán cảm xúc
    prob = model.predict_proba([features])[0]
    classes = model.classes_
    prob_percent = {cls: round(p * 100, 2) for cls, p in zip(classes, prob)}

    return jsonify({"prediction": prob_percent})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
