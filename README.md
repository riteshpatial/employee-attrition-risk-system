🚀 Employee Attrition Risk Prediction System

An end-to-end Machine Learning based web application that predicts employee attrition risk and visualizes insights using an interactive Streamlit dashboard.

This project covers the full data science lifecycle:
data analysis → feature engineering → model training → deployment-ready web app.

📌 Project Overview

This system helps organizations identify employees who are at high risk of leaving, so proactive retention strategies can be applied.

Key capabilities:

Upload employee dataset (CSV)

Predict attrition probability for each employee

Classify employees into High / Medium / Low risk

Department-level attrition analysis

Individual employee risk profiling

Feature importance (model explainability)

Downloadable risk report

🧠 Tech Stack

Python

Pandas, NumPy

Scikit-learn, Imbalanced-learn (SMOTE)

Matplotlib, Seaborn

Streamlit

Joblib

📂 Project Structure
employee-attrition-risk-system/
│
├── 01_attrition_model_training.ipynb   # Model training & experimentation
├── app.py                               # Streamlit web application
├── attrition_risk_model.pkl             # Trained ML pipeline
├── Palo Alto Networks.csv               # Sample dataset
├── requirements.txt                     # Project dependencies
└── README.md                            # Project documentation

⚙️ Installation & Setup

Clone the repository:

git clone https://github.com/riteshpatial/employee-attrition-risk-system.git
cd employee-attrition-risk-system


Create environment (recommended):

conda create -n attrition_env python=3.10
conda activate attrition_env


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py


App will open at:

http://localhost:8501

📊 Features Implemented

✔ Advanced feature engineering

✔ Class imbalance handling (SMOTE)

✔ ML pipeline with preprocessing + model

✔ Real-time prediction on uploaded CSV

✔ Dynamic risk threshold

✔ Department analytics

✔ High-risk employee detection

✔ Feature importance visualization

✔ Exportable prediction report

📈 Machine Learning Workflow

Data cleaning & EDA

Feature engineering:

EngagementScore

WorkStressScore

StabilityScore

IncomeExperienceRatio

Class imbalance handling (SMOTE)

Model training using Scikit-learn pipeline

Evaluation & tuning

Model serialization (joblib)

Deployment via Streamlit

🖥 Web App Preview

The Streamlit dashboard provides:

Dataset preview

Risk distribution charts

Department-wise attrition risk

Individual employee risk profile

High-risk employee table

Feature importance graph

🎯 Use Case

HR analytics

Workforce planning

Attrition prevention

People strategy optimization

👤 Author

Ritesh Patial
Data Analyst / ML Enthusiast

GitHub: https://github.com/riteshpatial

⚠ Disclaimer

This project is for educational and portfolio purposes. Predictions should not be used as the sole basis for HR decisions.

⭐ If you like this project

Give the repo a star and use it in your portfolio.
