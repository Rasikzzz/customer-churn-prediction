import joblib

model = joblib.load("models/churn_model.pkl")
features = joblib.load("models/features.pkl")

print("Model loaded successfully")
print("Number of features:", len(features))