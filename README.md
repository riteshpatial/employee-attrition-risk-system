# Employee Attrition Risk Prediction System

An end-to-end Machine Learning web application that predicts **which employees are at risk of leaving** — and by how much. Built with Scikit-learn + Streamlit, it covers the full data science lifecycle: data analysis → feature engineering → model training → interactive dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Dashboard Features](#dashboard-features)
- [Risk Classification](#risk-classification)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [How to Use](#how-to-use)
- [Use Cases](#use-cases)

---

## Project Overview

Employee attrition is one of the most costly problems for organizations. This system helps HR teams identify **high-risk employees before they resign**, so proactive retention strategies can be applied.

**What this system does:**

- Upload any employee CSV dataset
- Automatically engineer features and run ML predictions
- Classify each employee as **High / Medium / Low** attrition risk
- Show department-level risk analysis
- Profile individual employee risk
- Export a full risk report as CSV

---

## Project Structure

```
employee-attrition-risk-system/
│
├── 01_attrition_model_training.ipynb   # Full model training notebook (EDA → training → evaluation)
├── app.py                               # Streamlit web application
├── attrition_risk_model.pkl             # Trained ML pipeline (preprocessing + model)
├── Palo Alto Networks.csv               # Sample employee dataset
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

---

## Dataset

**File:** `Palo Alto Networks.csv`

The dataset contains **31 columns** covering employee demographics, job details, satisfaction scores, and work patterns.

| Column | Description |
|--------|-------------|
| `Age` | Employee age |
| `Attrition` | Target variable — 1 = Left, 0 = Stayed |
| `BusinessTravel` | Travel frequency (Rarely / Frequently / Non-Travel) |
| `Department` | Sales / Research & Development / Human Resources |
| `DistanceFromHome` | Commute distance (km) |
| `Education` | Education level (1–5) |
| `EnvironmentSatisfaction` | Workplace satisfaction (1–4) |
| `JobSatisfaction` | Job satisfaction score (1–4) |
| `JobLevel` | Seniority level (1–5) |
| `JobRole` | Role title (Sales Executive, Research Scientist, etc.) |
| `MonthlyIncome` | Monthly salary |
| `OverTime` | Yes / No |
| `PerformanceRating` | Performance score (1–4) |
| `RelationshipSatisfaction` | Peer/manager relationship score (1–4) |
| `TotalWorkingYears` | Total career experience |
| `WorkLifeBalance` | Work-life balance score (1–4) |
| `YearsAtCompany` | Tenure at current company |
| `YearsSinceLastPromotion` | Time since last promotion |
| `YearsWithCurrManager` | Years under current manager |
| `StockOptionLevel` | Stock option tier (0–3) |
| *...and more* | Demographics, rates, training data |

---

## Feature Engineering

Raw columns alone are not enough. These **7 custom features** are engineered before prediction:

| Feature | Formula | What It Captures |
|---------|---------|-----------------|
| `IncomeExperienceRatio` | `MonthlyIncome / TotalWorkingYears` | Is the employee underpaid for their experience? |
| `PromotionDelay` | `YearsSinceLastPromotion / (YearsAtCompany + 1)` | How long since their last growth opportunity? |
| `EngagementScore` | `(JobSatisfaction + EnvironmentSatisfaction + RelationshipSatisfaction + WorkLifeBalance) / 4` | Overall employee engagement level |
| `WorkStressScore` | `OverTime + DistanceFromHome + FrequentTravel` | Composite work stress indicator |
| `StabilityScore` | `YearsWithCurrManager + YearsAtCompany + TotalWorkingYears` | How rooted is the employee? |
| `WorkloadStressFlag` | `1 if OverTime=Yes AND WorkLifeBalance ≤ 2` | Burnout warning flag |

---

## Machine Learning Pipeline

```
Raw CSV
   ↓
Feature Engineering (6 custom features added)
   ↓
Preprocessing Pipeline (Scikit-learn)
   ├── OneHotEncoder → Categorical columns
   └── StandardScaler → Numerical columns
   ↓
SMOTE (handle class imbalance — fewer people leave than stay)
   ↓
Model Training (Random Forest / Tree-based classifier)
   ↓
Probability Output → AttritionProbability (0.0 – 1.0)
   ↓
Risk Classification → High / Medium / Low
   ↓
Serialized Pipeline → attrition_risk_model.pkl
```

**Training notebook:** `01_attrition_model_training.ipynb`
- Full EDA with charts
- Class imbalance analysis
- SMOTE application
- Model evaluation (accuracy, precision, recall, F1, ROC-AUC)
- Feature importance analysis

---

## Dashboard Features

The Streamlit app (`app.py`) has 7 sections:

| Section | What It Shows |
|---------|--------------|
| **Dataset Preview** | First 5 rows of uploaded file |
| **Risk Distribution** | Bar chart — how many High / Medium / Low risk employees |
| **KPI Cards** | Count of High, Medium, Low risk employees at a glance |
| **Department Analysis** | Average attrition risk per department (bar chart + table) |
| **Employee Profile** | Select any employee → see their probability + all features |
| **High Risk Table** | All high-risk employees sorted by probability (descending) |
| **Feature Importance** | Top 15 features driving the model's decisions |
| **Download Report** | Full dataset with risk scores as CSV |

**Sidebar controls:**
- CSV file uploader
- Risk threshold slider (0.3 – 0.9) — adjust what counts as "High Risk"

---

## Risk Classification

| Risk Level | Condition | Meaning |
|------------|-----------|---------|
| **High Risk** | Probability ≥ threshold (default: 0.6) | Likely to leave — take action now |
| **Medium Risk** | 0.3 ≤ Probability < threshold | Monitor closely |
| **Low Risk** | Probability < 0.3 | Stable employee |

> The threshold is adjustable in real-time using the sidebar slider — HR teams can tune sensitivity based on their needs.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | Interactive web dashboard |
| `scikit-learn` | ML pipeline, preprocessing, model training |
| `imbalanced-learn` | SMOTE for class imbalance |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Feature importance plots |
| `joblib` | Model serialization / loading |
| `pyarrow` | Fast CSV handling in Streamlit |

---

## Setup & Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/riteshpatial/employee-attrition-risk-system.git
cd employee-attrition-risk-system
```

### Step 2 — Create Virtual Environment (Recommended)

**Using conda:**
```bash
conda create -n attrition_env python=3.10
conda activate attrition_env
```

**Using venv:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the App

```bash
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## How to Use

1. Open the app at `http://localhost:8501`
2. In the **sidebar**, upload your employee CSV file
3. Adjust the **High Risk Threshold** slider if needed (default: 0.6)
4. View the automatically generated dashboard:
   - Overall risk distribution
   - Department-level risk heatmap
   - Individual employee profile
   - High-risk employee list
5. Click **Download Risk Report** to export results as CSV

> **Sample file:** Use `Palo Alto Networks.csv` to test the app immediately after setup.

---

## Use Cases

| Team | How They Use It |
|------|----------------|
| **HR Analytics** | Identify at-risk employees before they resign |
| **Workforce Planning** | Plan hiring pipeline based on predicted attrition |
| **People Strategy** | Design targeted retention programs per department |
| **Management** | Flag teams with high WorkStressScore or low EngagementScore |

---

## Author

**Ritesh Patial** — Data Analyst / ML Engineer

GitHub: [github.com/riteshpatial](https://github.com/riteshpatial)

---

> **Disclaimer:** This project is for educational and portfolio purposes. Model predictions should support — not replace — human HR judgment.
