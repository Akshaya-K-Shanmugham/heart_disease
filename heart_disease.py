import pymongo

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Change if using Atlas

# Select the heart_disease_db database
db = client["heart_disease_db"]

# Select the collection "patients" within the heart_disease_db database
collection = db["patients"]

# Now you can work with this collection, for example, inserting or querying data
import pymongo

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Change if using Atlas

# Select or create the heart_disease_db database
db = client["heart_disease_db"]

# Select the 'patients' collection
collection = db["patients"]

# Insert example data
patients_data = [
    { "age": 50, "sex": 1, "blood_pressure": 140, "cholesterol": 230, "heart_disease": 1 },
    { "age": 60, "sex": 0, "blood_pressure": 130, "cholesterol": 210, "heart_disease": 0 },
    { "age": 45, "sex": 1, "blood_pressure": 120, "cholesterol": 190, "heart_disease": 0 },
    { "age": 65, "sex": 0, "blood_pressure": 150, "cholesterol": 240, "heart_disease": 1 }
]

# Insert the data into the collection
collection.insert_many(patients_data)

print("Data inserted successfully!")

import pymongo
import pandas as pd

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Change if using Atlas

# Select the heart_disease_db database
db = client["heart_disease_db"]

# Select the patients collection
collection = db["patients"]

# Fetch all documents from the collection and convert them to a pandas DataFrame
data = pd.DataFrame(list(collection.find()))

# Display the data
print(data)

# Check for missing values
print(data.isnull().sum())

# Assuming no missing values, we proceed to feature engineering

# Drop non-essential columns (e.g., _id)
data = data.drop(columns=["_id"])

# Separate the features (X) and the target variable (y)
X = data.drop(columns=["heart_disease"])  # Features
y = data["heart_disease"]  # Target variable

# Normalize or scale the data (optional, but often necessary for better performance)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])  # Display the scaled features

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize the classifier
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))
import joblib

# Save the model to a file
joblib.dump(model, "heart_disease_model.pkl")

# Save the scaler as well for future predictions
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")
# Load the saved model and scaler
loaded_model = joblib.load("heart_disease_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

# New data for prediction (example)
new_data = pd.DataFrame({
    "age": [55],
    "sex": [1],  # male
    "blood_pressure": [130],
    "cholesterol": [200]
})

# Scale the new data using the loaded scaler
new_data_scaled = loaded_scaler.transform(new_data)

# Make the prediction
prediction = loaded_model.predict(new_data_scaled)
print("Prediction for new data:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")



