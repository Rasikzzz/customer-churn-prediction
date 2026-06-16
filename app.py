import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load model + features
# -----------------------------
model = joblib.load("models/churn_model.pkl")
model_features = joblib.load("models/features.pkl")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Explainable Churn AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Explainable Customer Churn AI System")
st.markdown("ML + XGBoost + SHAP Explainability Dashboard")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

# -----------------------------
# Build input dataframe
# -----------------------------
input_dict = {
    "SeniorCitizen": [1 if senior == "Yes" else 0],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
}

input_df = pd.DataFrame(input_dict)

# Encode categorical features (must match training)
input_df["gender_Male"] = 1 if gender == "Male" else 0
input_df["Partner_Yes"] = 1 if partner == "Yes" else 0
input_df["Dependents_Yes"] = 1 if dependents == "Yes" else 0

input_df["Contract_One year"] = 1 if contract == "One year" else 0
input_df["Contract_Two year"] = 1 if contract == "Two year" else 0

input_df["InternetService_Fiber optic"] = 1 if internet_service == "Fiber optic" else 0
input_df["InternetService_No"] = 1 if internet_service == "No" else 0

# Align features
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]

# -----------------------------
# Prediction Section
# -----------------------------
if st.button("Predict Churn Risk"):
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ Customer WILL CHURN")
        else:
            st.success("✅ Customer WILL STAY")

    with col2:
        st.metric("Churn Probability", f"{probability:.2%}")

    # Risk level
    st.subheader("🚦 Risk Level")

    if probability > 0.7:
        st.error("🔴 High Risk Customer")
    elif probability > 0.4:
        st.warning("🟡 Medium Risk Customer")
    else:
        st.success("🟢 Low Risk Customer")

    # -----------------------------
    # SHAP Explainability
    # -----------------------------
    st.subheader("🧠 Why this prediction? (SHAP Explainability)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns
        ),
        show=False
    )

    st.pyplot(fig)

    # -----------------------------
    # Business Explanation
    # -----------------------------
    st.subheader("💡 Business Insights")

    st.markdown("""
    - SHAP shows which features increased or decreased churn risk.
    - Positive values → increase churn probability.
    - Negative values → reduce churn probability.
    """)
