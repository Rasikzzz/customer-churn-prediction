# 📊 Dataset Information

## 📌 Overview
This folder provides information about the dataset used in the Customer Churn Prediction project. The dataset contains telecom customer data used to predict customer churn (whether a customer will leave the service or not).

---

## 📂 Dataset Used

**IBM Telco Customer Churn Dataset**

- Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
- Type: Tabular Dataset  
- Problem Type: Binary Classification (Churn / No Churn)

---

## ⚙️ Data Access Method

The dataset is not manually stored in this repository. It is automatically downloaded using `kagglehub`.

### Code Used:

```python
import kagglehub

# Download latest version of dataset
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("Dataset path:", path)
