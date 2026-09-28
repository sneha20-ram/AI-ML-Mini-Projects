# 💳 Explainable ML Credit Risk Decision System

An educational machine learning application that predicts credit risk from applicant information using an XGBoost classification model.

The project includes model comparison, preprocessing, prediction probability, and interactive What-If analysis through a Streamlit web application.

> **Note:** This project is for educational purposes only and should not be used for real financial decisions.

---

## 🚀 Features

- Credit risk prediction using XGBoost
- Logistic Regression baseline for comparison
- Data preprocessing using Scikit-learn
- Categorical feature encoding using OneHotEncoder
- Numerical feature scaling using StandardScaler
- Risk probability prediction
- Interactive Streamlit interface
- What-If analysis for:
  - Loan duration
  - Credit amount
- Modular prediction pipeline
- Reproducible project structure

---

## 🧠 Machine Learning Models

Two classification models were evaluated:

### Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 0.780 |
| Precision | 0.667 |
| Recall | 0.533 |
| F1 Score | 0.593 |
| ROC-AUC | 0.804 |

### XGBoost

| Metric | Score |
|---|---:|
| Accuracy | 0.765 |
| Precision | 0.644 |
| Recall | 0.483 |
| F1 Score | 0.552 |
| ROC-AUC | 0.799 |

The XGBoost model is used in the final Streamlit application.

---

## 📊 Dataset

The project uses the **UCI Statlog German Credit Dataset**.

Dataset characteristics:

- 1,000 instances
- 20 input attributes
- Binary classification target
- Mixture of numerical and categorical features

The target is mapped as:

```text
0 → Lower Risk / Good Credit
1 → Higher Risk / Bad Credit