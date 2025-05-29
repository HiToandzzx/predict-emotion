# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv('data/dataset.csv')

# ✅ Filter only dogs
df = df[df['Animal_Type'] == 'Dog']

# Keep needed columns
columns_to_keep = [
    'Breed', 'Gender', 'Age', 'Weight', 'Duration',
    'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing', 'Labored_Breathing',
    'Lameness', 'Skin_Lesions', 'Nasal_Discharge', 'Eye_Discharge',
    'Body_Temperature', 'Heart_Rate', 'Disease_Prediction'
]
df = df[columns_to_keep]

# ✅ Convert binary features: Yes/No to 1/0
binary_cols = ['Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
               'Labored_Breathing', 'Lameness', 'Skin_Lesions',
               'Nasal_Discharge', 'Eye_Discharge']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# ✅ Clean Body_Temperature
df['Body_Temperature'] = df['Body_Temperature'].str.extract(r'(\d+\.?\d*)').astype(float)

# ✅ Encode categorical columns
label_encoders = {}
categorical_cols = ['Breed', 'Gender', 'Duration', 'Disease_Prediction']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features and target
X = df.drop(columns=['Disease_Prediction'])
y = df['Disease_Prediction']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save all encoders (dictionary of LabelEncoders)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ Model and encoders saved.")