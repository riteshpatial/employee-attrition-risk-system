import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Employee Attrition Risk System",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2942 100%);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.08);
  }
  .main-header h1 {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 6px 0;
  }
  .main-header p { color: rgba(255,255,255,0.65); font-size: .9rem; margin: 0; }

  .kpi-card {
    padding: 20px 24px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid;
  }
  .kpi-card .kpi-val { font-size: 2.2rem; font-weight: 800; line-height: 1; margin-bottom: 6px; }
  .kpi-card .kpi-label { font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; opacity: 0.75; }

  .kpi-high   { background: #fff1f2; border-color: #fecaca; }
  .kpi-high   .kpi-val { color: #dc2626; }
  .kpi-high   .kpi-label { color: #dc2626; }

  .kpi-medium { background: #fffbeb; border-color: #fde68a; }
  .kpi-medium .kpi-val { color: #d97706; }
  .kpi-medium .kpi-label { color: #d97706; }

  .kpi-low    { background: #f0fdf4; border-color: #bbf7d0; }
  .kpi-low    .kpi-val { color: #16a34a; }
  .kpi-low    .kpi-label { color: #16a34a; }

  .kpi-total  { background: #eff6ff; border-color: #bfdbfe; }
  .kpi-total  .kpi-val { color: #1d4ed8; }
  .kpi-total  .kpi-label { color: #1d4ed8; }

  .section-title {
    font-size: 1.1rem; font-weight: 700; color: #1e293b;
    margin: 32px 0 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
  }

  .risk-badge-high   { background:#fee2e2; color:#dc2626; padding:3px 10px; border-radius:20px; font-weight:700; font-size:.78rem; }
  .risk-badge-medium { background:#fef3c7; color:#d97706; padding:3px 10px; border-radius:20px; font-weight:700; font-size:.78rem; }
  .risk-badge-low    { background:#dcfce7; color:#16a34a; padding:3px 10px; border-radius:20px; font-weight:700; font-size:.78rem; }

  div[data-testid="stSidebar"] { background: #0f172a !important; }
  div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  div[data-testid="stSidebar"] .stSlider > div > div > div { background: #334155 !important; }

  .stDownloadButton > button {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 10px 24px !important; width: 100%;
  }

  .demo-banner {
    background: linear-gradient(90deg, #0f172a, #1e3a5f);
    color: #93c5fd; padding: 10px 18px; border-radius: 10px;
    font-size: .85rem; font-weight: 500; margin-bottom: 20px;
    border: 1px solid #1e40af;
  }
</style>
""", unsafe_allow_html=True)

# ── Load Model ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("attrition_risk_model.pkl")

model = load_model()

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Data Source")
    uploaded_file = st.file_uploader("Upload Employee CSV", type=["csv"])
    st.markdown("<div style='text-align:center;color:#64748b;margin:8px 0'>— or —</div>", unsafe_allow_html=True)
    use_sample = st.button("▶ Load Sample Data (Demo)", use_container_width=True)
    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    risk_threshold = st.slider("High Risk Threshold", 0.3, 0.9, 0.6, 0.05,
                               help="Probability above this = High Risk")
    st.markdown("---")
    st.markdown("<div style='font-size:.75rem;color:#475569;text-align:center'>Built by Ritesh Patial<br>ML · HR Analytics</div>", unsafe_allow_html=True)

if use_sample:
    st.session_state["use_sample"] = True
if uploaded_file:
    st.session_state["use_sample"] = False

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📊 Employee Attrition Risk Prediction</h1>
  <p>ML-powered system to identify at-risk employees before they resign · Random Forest · SMOTE · Feature Engineering</p>
</div>
""", unsafe_allow_html=True)

# ── Main ─────────────────────────────────────────────────────────────
if uploaded_file or st.session_state.get("use_sample"):

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("Palo Alto Networks.csv")
        st.markdown('<div class="demo-banner">📌 Demo Mode — Palo Alto Networks sample dataset loaded. Upload your own CSV to analyze real data.</div>', unsafe_allow_html=True)

    if "Attrition" in df.columns:
        df = df.drop(columns=["Attrition"])

    # Feature Engineering
    df["TotalWorkingYears"] = df["TotalWorkingYears"].replace(0, 1)
    df["IncomeExperienceRatio"] = df["MonthlyIncome"] / df["TotalWorkingYears"]
    df["PromotionDelay"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["EngagementScore"] = (df["JobSatisfaction"] + df["EnvironmentSatisfaction"] + df["RelationshipSatisfaction"] + df["WorkLifeBalance"]) / 4
    df["WorkStressScore"] = ((df["OverTime"] == "Yes").astype(int) + df["DistanceFromHome"] + (df["BusinessTravel"] == "Travel_Frequently").astype(int))
    df["StabilityScore"] = df["YearsWithCurrManager"] + df["YearsAtCompany"] + df["TotalWorkingYears"]
    df["WorkloadStressFlag"] = ((df["OverTime"] == "Yes") & (df["WorkLifeBalance"] <= 2)).astype(int)

    # Predictions
    preds = model.predict_proba(df)[:, 1]
    df["AttritionProbability"] = preds

    def risk_label(p):
        if p >= risk_threshold: return "High Risk"
        elif p >= 0.3:          return "Medium Risk"
        else:                   return "Low Risk"

    df["RiskCategory"] = df["AttritionProbability"].apply(risk_label)

    high   = (df["RiskCategory"] == "High Risk").sum()
    medium = (df["RiskCategory"] == "Medium Risk").sum()
    low    = (df["RiskCategory"] == "Low Risk").sum()
    total  = len(df)

    # ── KPI Cards ──
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card kpi-total"><div class="kpi-val">{total}</div><div class="kpi-label">Total Employees</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card kpi-high"><div class="kpi-val">{high}</div><div class="kpi-label">High Risk</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card kpi-medium"><div class="kpi-val">{medium}</div><div class="kpi-label">Medium Risk</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card kpi-low"><div class="kpi-val">{low}</div><div class="kpi-label">Low Risk</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row ──
    st.markdown('<div class="section-title">Risk Analysis</div>', unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        fig_pie = px.pie(
            values=[high, medium, low],
            names=["High Risk", "Medium Risk", "Low Risk"],
            color_discrete_sequence=["#dc2626", "#d97706", "#16a34a"],
            hole=0.5,
            title="Risk Distribution"
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter", title_font_size=14, margin=dict(t=40, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch2:
        dept_risk = df.groupby("Department")["AttritionProbability"].mean().reset_index()
        dept_risk.columns = ["Department", "Avg Risk"]
        dept_risk = dept_risk.sort_values("Avg Risk", ascending=True)
        fig_dept = px.bar(
            dept_risk, x="Avg Risk", y="Department", orientation="h",
            color="Avg Risk", color_continuous_scale=["#16a34a", "#d97706", "#dc2626"],
            title="Avg Attrition Risk by Department", text=dept_risk["Avg Risk"].round(2)
        )
        fig_dept.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter", title_font_size=14, margin=dict(t=40, b=10, l=10, r=10),
            coloraxis_showscale=False, xaxis_title="", yaxis_title=""
        )
        fig_dept.update_traces(textposition="outside")
        st.plotly_chart(fig_dept, use_container_width=True)

    # ── High Risk Table ──
    st.markdown('<div class="section-title">High Risk Employees</div>', unsafe_allow_html=True)
    high_df = df[df["RiskCategory"] == "High Risk"].sort_values("AttritionProbability", ascending=False)
    display_cols = ["JobRole", "Department", "MonthlyIncome", "YearsAtCompany", "OverTime", "AttritionProbability", "RiskCategory"]
    available = [c for c in display_cols if c in high_df.columns]
    st.dataframe(
        high_df[available].style.background_gradient(subset=["AttritionProbability"], cmap="Reds"),
        use_container_width=True, height=320
    )

    # ── Employee Profile ──
    st.markdown('<div class="section-title">Employee Risk Profile</div>', unsafe_allow_html=True)
    p1, p2 = st.columns([1, 2])
    with p1:
        emp_id = st.selectbox("Select Employee", df.index, format_func=lambda x: f"Employee #{x}")
        emp = df.loc[emp_id]
        risk = emp["RiskCategory"]
        prob = round(emp["AttritionProbability"] * 100, 1)
        badge_class = "high" if risk == "High Risk" else "medium" if risk == "Medium Risk" else "low"
        color = "#dc2626" if risk == "High Risk" else "#d97706" if risk == "Medium Risk" else "#16a34a"
        st.markdown(f"""
        <div style="background:#f8fafc;padding:20px;border-radius:14px;border:1px solid #e2e8f0;text-align:center;margin-top:10px">
          <div style="font-size:2.8rem;font-weight:800;color:{color}">{prob}%</div>
          <div style="font-size:.8rem;color:#64748b;margin:4px 0 10px">Attrition Probability</div>
          <span class="risk-badge-{badge_class}">{risk}</span>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        key_fields = ["JobRole", "Department", "MonthlyIncome", "YearsAtCompany",
                      "OverTime", "JobSatisfaction", "WorkLifeBalance", "EngagementScore", "WorkStressScore"]
        available_fields = [f for f in key_fields if f in emp.index]
        profile_data = pd.DataFrame({"Field": available_fields, "Value": [str(emp[f]) for f in available_fields]})
        st.dataframe(profile_data, use_container_width=True, hide_index=True, height=280)

    # ── Feature Importance ──
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    if hasattr(model[-1], "feature_importances_"):
        importances = model[-1].feature_importances_
        features = model[:-1].get_feature_names_out()
        fi = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=True).tail(15)
        fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale=["#bfdbfe", "#1d4ed8"],
                        title="Top 15 Features Driving Predictions")
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter", title_font_size=14, margin=dict(t=40, b=10),
            coloraxis_showscale=False, xaxis_title="", yaxis_title=""
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Download ──
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Full Risk Report (CSV)", csv, "attrition_risk_report.csv", "text/csv")

else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px">
      <div style="font-size:3.5rem;margin-bottom:16px">📊</div>
      <h3 style="color:#1e293b;font-size:1.4rem;margin-bottom:10px">No Data Loaded</h3>
      <p style="color:#64748b;max-width:420px;margin:0 auto">Upload an employee CSV file or click <strong>Load Sample Data</strong> in the sidebar to see the dashboard in action.</p>
    </div>
    """, unsafe_allow_html=True)
