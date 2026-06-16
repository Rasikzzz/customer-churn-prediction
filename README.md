# 📊 Customer Churn Prediction System

An end-to-end Machine Learning project that predicts customer churn using **XGBoost** and provides a real-time interactive web application using **Streamlit**.

The system analyzes customer behavior and identifies whether a customer is likely to leave the service, helping businesses take proactive retention actions.

---

## 📌 Problem Statement

Customer churn is a major problem in subscription-based businesses. Retaining existing customers is more cost-effective than acquiring new ones. This project aims to predict churn probability and identify key factors influencing customer decisions.

---

## 🧠 Solution Approach

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering and encoding
- Model training using **XGBoost Classifier**
- Model evaluation using multiple metrics
- Deployment using **Streamlit web app**

---

## 📊 Dataset

IBM Telco Customer Churn Dataset  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## 🛠️ Tech Stack

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib

---

## 📈 Model Performance

- Accuracy: ~79.8%
- Precision: ~64.6%
- Recall: ~53.2%
- F1 Score: ~58.4%
- ROC-AUC: ~84%

---

## 📁 Project Structure

```text id="rme4"
customer-churn-prediction/
│
├── app.py                  # Streamlit web app
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
│
├── models/
│   ├── churn_model.pkl     # Trained XGBoost model
│   └── features.pkl        # Feature list
│
├── data/
│   └── README.md           # Dataset information
│
├── notebooks/
│   └── exploration.ipynb   # EDA and model training
│
└── assets/
    └── demo.mp4            # Screen recording (optional)
