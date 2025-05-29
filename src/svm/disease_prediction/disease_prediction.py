# disease_prediction.py
import pickle
import pandas as pd

# Load model and encoders
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Sample input: one record (same columns as used in training)
input_data = {
    'Breed': 'Beagle',
    'Gender': 'Male',
    'Age': 2,
    'Weight': 10,
    'Duration': '4 days',
    'Appetite_Loss': 'Yes',
    'Vomiting': 'Yes',
    'Diarrhea': 'Yes',
    'Coughing': 'Yes',
    'Labored_Breathing': 'No',
    'Lameness': 'Yes',
    'Skin_Lesions': 'Yes',
    'Nasal_Discharge': 'No',
    'Eye_Discharge': 'Yes',
    'Body_Temperature': '39.4°C',
    'Heart_Rate': 120
}

# Convert to DataFrame
df_input = pd.DataFrame([input_data])

# Preprocessing
binary_cols = ['Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
               'Labored_Breathing', 'Lameness', 'Skin_Lesions',
               'Nasal_Discharge', 'Eye_Discharge']
for col in binary_cols:
    df_input[col] = df_input[col].map({'Yes': 1, 'No': 0})

# Clean temperature
df_input['Body_Temperature'] = df_input['Body_Temperature'].str.extract(r'(\d+\.?\d*)').astype(float)

# Encode categorical columns
for col in ['Breed', 'Gender', 'Duration']:
    le = label_encoders[col]
    df_input[col] = le.transform(df_input[col])

# Predict
prediction_encoded = model.predict(df_input)[0]

# Decode predicted label
disease_le = label_encoders['Disease_Prediction']
predicted_disease = disease_le.inverse_transform([prediction_encoded])[0]

print(f"✅ Predicted Disease: {predicted_disease}")