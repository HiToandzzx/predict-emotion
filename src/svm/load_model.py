from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image
import pickle
import pandas as pd


app = Flask(__name__)

# Load emotion prediction model
model_path = "emotion_prediction/svm_dog_emotion_model.pkl"
model = joblib.load(model_path)

# Disease prediction model paths
disease_model_path = "disease_prediction/svm_model.pkl"
label_encoders_path = "disease_prediction/label_encoders.pkl"

# Load disease prediction model
with open(disease_model_path, 'rb') as f:
    disease_model = pickle.load(f)
with open(label_encoders_path, 'rb') as f:
    disease_label_encoders = pickle.load(f)
print("Disease model loaded successfully.")


def preprocess_disease_input(input_data):
    """Preprocess input data for disease prediction"""
    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])

    # Convert binary features
    binary_cols = ['Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
                   'Labored_Breathing', 'Lameness', 'Skin_Lesions',
                   'Nasal_Discharge', 'Eye_Discharge']
    for col in binary_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].map({'Yes': 1, 'No': 0})

    # Clean temperature
    if 'Body_Temperature' in df_input.columns:
        df_input['Body_Temperature'] = df_input['Body_Temperature'].astype(str).str.extract(r'(\d+\.?\d*)').astype(
            float)

    # Encode categorical columns
    for col in ['Breed', 'Gender', 'Duration']:
        if col in df_input.columns and col in disease_label_encoders:
            le = disease_label_encoders[col]
            try:
                df_input[col] = le.transform(df_input[col])
            except ValueError as e:
                # Handle unknown categories
                print(f"Unknown category in {col}: {df_input[col].values[0]}")
                # Use the most frequent class as default
                df_input[col] = 0

    return df_input


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
    return jsonify({
        "message": "Welcome to Dog Prediction API",
        "endpoints": {
            "/predict": "POST - Dog Emotion Prediction (upload image)",
            "/predict_disease": "POST - Dog Disease Prediction (send JSON data)"
        },
        "available_services": [
            "Dog Emotion Recognition from Images",
            "Dog Disease Prediction from Symptoms"
        ]
    })


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


# API dự đoán bệnh
@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Required fields for disease prediction
        required_fields = [
            'Breed', 'Gender', 'Age', 'Weight', 'Duration',
            'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing', 'Labored_Breathing',
            'Lameness', 'Skin_Lesions', 'Nasal_Discharge', 'Eye_Discharge',
            'Body_Temperature', 'Heart_Rate'
        ]

        # Check if all required fields are present
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        # Preprocess the input data
        processed_data = preprocess_disease_input(data)

        # Make prediction
        #prediction_encoded = disease_model.predict(processed_data)[0]
        prediction_proba = disease_model.predict_proba(processed_data)[0]

        # Get index of max probability
        max_index = np.argmax(prediction_proba)

        # Decode predicted label from max index
        disease_le = disease_label_encoders['Disease_Prediction']
        predicted_disease = disease_le.inverse_transform([max_index])[0]

        # Return only predicted disease and confidence
        return jsonify({
            "predicted_disease": predicted_disease,
            "confidence": round(prediction_proba[max_index] * 100, 2),
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)