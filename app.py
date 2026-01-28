import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Employee Attrition Risk System",
    layout="wide",
    page_icon="📊"
)

st.title("Employee Attrition Risk Prediction System")

# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = "attrition_risk_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Upload Employee Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
risk_threshold = st.sidebar.slider("High Risk Threshold", 0.3, 0.9, 0.6)

# ===============================
# MAIN
# ===============================
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully")

    st.subheader("Preview of Dataset")
    st.dataframe(df.head(), use_container_width=True)

    if "Attrition" in df.columns:
        df = df.drop(columns=["Attrition"])

    # ===============================
    # FEATURE ENGINEERING (MATCH TRAINING)
    # ===============================
    df["TotalWorkingYears"] = df["TotalWorkingYears"].replace(0, 1)

    df["IncomeExperienceRatio"] = df["MonthlyIncome"] / df["TotalWorkingYears"]

    df["PromotionDelay"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)

    df["EngagementScore"] = (
        df["JobSatisfaction"] +
        df["EnvironmentSatisfaction"] +
        df["RelationshipSatisfaction"] +
        df["WorkLifeBalance"]
    ) / 4

    df["WorkStressScore"] = (
        (df["OverTime"] == "Yes").astype(int) +
        df["DistanceFromHome"] +
        (df["BusinessTravel"] == "Travel_Frequently").astype(int)
    )

    df["StabilityScore"] = (
        df["YearsWithCurrManager"] +
        df["YearsAtCompany"] +
        df["TotalWorkingYears"]
    )

    df["WorkloadStressFlag"] = (
        (df["OverTime"] == "Yes") & (df["WorkLifeBalance"] <= 2)
    ).astype(int)

    # ===============================
    # PREDICTION
    # ===============================
    preds = model.predict_proba(df)[:, 1]
    df["AttritionProbability"] = preds

    def risk_label(p):
        if p >= risk_threshold:
            return "High Risk"
        elif p >= 0.3:
            return "Medium Risk"
        else:
            return "Low Risk"

    df["RiskCategory"] = df["AttritionProbability"].apply(risk_label)

    # ===============================
    # OVERALL DASHBOARD
    # ===============================
    st.subheader("Overall Risk Distribution")
    st.bar_chart(df["RiskCategory"].value_counts())

    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk", (df["RiskCategory"] == "High Risk").sum())
    col2.metric("Medium Risk", (df["RiskCategory"] == "Medium Risk").sum())
    col3.metric("Low Risk", (df["RiskCategory"] == "Low Risk").sum())

    # ===============================
    # DEPARTMENT VIEW
    # ===============================
    st.subheader("Department Level Attrition Risk")
    dept_risk = df.groupby("Department")["AttritionProbability"].mean().reset_index()
    st.bar_chart(dept_risk.set_index("Department"))
    st.dataframe(dept_risk.rename(columns={"AttritionProbability": "Average Risk"}), use_container_width=True)

    # ===============================
    # EMPLOYEE PROFILE  (FIXED)
    # ===============================
    st.subheader("Employee Risk Profile")
    emp_id = st.selectbox("Select Employee Index", df.index)
    emp = df.loc[emp_id]

    col1, col2 = st.columns(2)
    col1.metric("Attrition Probability", round(emp["AttritionProbability"], 3))
    col2.metric("Risk Category", emp["RiskCategory"])

    profile_df = emp.drop(["AttritionProbability", "RiskCategory"]).reset_index()
    profile_df.columns = ["Feature", "Value"]
    profile_df["Value"] = profile_df["Value"].astype(str)

    st.dataframe(profile_df, use_container_width=True)

    # ===============================
    # HIGH RISK EMPLOYEES
    # ===============================
    st.subheader("High Risk Employees")

    high_risk_df = df[df["RiskCategory"] == "High Risk"] \
        .sort_values("AttritionProbability", ascending=False)

    st.dataframe(high_risk_df, use_container_width=True)

    # ===============================
    # MODEL EXPLAINABILITY
    # ===============================
    st.subheader("Model Explainability (Feature Importance)")

    if hasattr(model[-1], "feature_importances_"):
        importances = model[-1].feature_importances_
        features = model[:-1].get_feature_names_out()

        fi = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(fi["Feature"], fi["Importance"])
        ax.invert_yaxis()
        st.pyplot(fig)
    else:
        st.info("Explainability supported only for tree-based models.")

    # ===============================
    # FULL DATASET
    # ===============================
    st.subheader("Full Dataset with Risk Scores")
    st.dataframe(df, use_container_width=True)

    # ===============================
    # DOWNLOAD
    # ===============================
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Risk Report",
        csv,
        "employee_attrition_risk_report.csv",
        "text/csv"
    )

else:
    st.info("Upload employee dataset to begin.")
