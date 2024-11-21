import pymongo
client = pymongo.MongoClient("mongodb://localhost:27017/") 
db = client["heart_disease_db"]
collection = db["patients"]

patients_data = [
    { "age": 50, "sex": 1, "blood_pressure": 140, "cholesterol": 230, "heart_disease": 1 },
    { "age": 60, "sex": 0, "blood_pressure": 130, "cholesterol": 210, "heart_disease": 0 },
    { "age": 45, "sex": 1, "blood_pressure": 120, "cholesterol": 190, "heart_disease": 0 },
    { "age": 65, "sex": 0, "blood_pressure": 150, "cholesterol": 240, "heart_disease": 1 }
]
collection.insert_many(patients_data)
print("Data inserted successfully!")
data = pd.DataFrame(list(collection.find()))
print(data)
print(data.isnull().sum())
data = data.drop(columns=["_id"])


X = data.drop(columns=["heart_disease"])  
y = data["heart_disease"] 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5]) 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

import joblib
joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")
loaded_model = joblib.load("heart_disease_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
new_data = pd.DataFrame({
    "age": [55],
    "sex": [1],  # male
    "blood_pressure": [130],
    "cholesterol": [200]
})
new_data_scaled = loaded_scaler.transform(new_data)
prediction = loaded_model.predict(new_data_scaled)
print("Prediction for new data:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")



