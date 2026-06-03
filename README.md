# 🧠 DASS-42 Mental Health Analysis Dashboard

An interactive data science project analyzing Depression, Anxiety, and Stress patterns using the DASS-42 psychological assessment dataset — featuring exploratory data analysis, machine learning prediction, and a deployed Streamlit dashboard.

---

## 🌐 Live Dashboard

[Open Dashboard](https://mental-health-dashboard-7wpzusncyc7nkxkdioafwo.streamlit.app)

---

## 📌 Project Overview

This project follows a complete end-to-end data science pipeline:

| Stage | Description |
|---|---|
| Data Collection | DASS-42 survey — 38,025 participants, 172 features |
| EDA & Cleaning | Feature selection, outlier removal, KNN imputation |
| Feature Engineering | DASS scores, severity levels, TIPI composites, age groups |
| ML Modeling | 8 models compared, XGBoost selected, 90%+ accuracy |
| Dashboard | 5-section Streamlit app deployed on Streamlit Cloud |

---

## 📊 Dashboard Sections

- **ML Prediction** — Answer 42 DASS questions + demographics + personality to get predicted Depression, Anxiety, and Stress severity levels
- **Overview** — KPI cards, level distributions, correlation matrix, high-risk counts
- **Demographics** — Mental health scores across age groups, gender, and education
- **Psychological Insights** — Strongest symptom indicators ranked by average severity
- **High-Risk Insights** — High-risk participant analysis by condition, gender, and age group

---

## 🤖 Machine Learning

### Approach
- 8 algorithms compared using 5-Fold Stratified Cross-Validation
- XGBoost selected as best performer across all 3 targets
- GridSearchCV hyperparameter tuning applied

### Results

| Target | Model | Test Accuracy | CV Mean |
|---|---|---|---|
| Depression | XGBoost | 89.9% | 89.2% |
| Anxiety | XGBoost | 91.2% | 90.8% |
| Stress | XGBoost | 92.1% | 91.6% |

### Feature Set
- All 42 DASS question responses (Q1A–Q42A)
- 5 demographic features (age, gender, education, family size, marital status)
- 5 TIPI personality composites (Emotional Stability, Extraversion, Agreeableness, Conscientiousness, Openness)

mental-health-dashboard/
├── app.py                      # Streamlit dashboard
├── data_cleaned.csv            # Cleaned DASS dataset
├── question_categories.csv     # Question-symptom mapping
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml             # Light mode theme config
├── models/
│   ├── model_depression.pkl    # Trained XGBoost — Depression
│   ├── model_anxiety.pkl       # Trained XGBoost — Anxiety
│   ├── model_stress.pkl        # Trained XGBoost — Stress
│   ├── encoder_depression.pkl  # Label encoder — Depression
│   ├── encoder_anxiety.pkl     # Label encoder — Anxiety
│   ├── encoder_stress.pkl      # Label encoder — Stress
│   ├── features_depression.pkl # Feature list — Depression
│   ├── features_anxiety.pkl    # Feature list — Anxiety
│   └── features_stress.pkl     # Feature list — Stress
└── *.png                       # Dashboard icons


---

## 🗂️ Dataset

- **Source:** [DASS-42 — OpenPsychometrics](https://openpsychometrics.org/datasets/)
- **Participants:** 38,025
- **Original Features:** 172
- **Features After Cleaning:** 67
- **Target Variables:** depression_level, anxiety_level, stress_level
- **Classes:** Low · Moderate · High

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/ALAAMEKAWY56/mental-health-dashboard.git
cd mental-health-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## 📦 Requirements

streamlit
pandas
numpy
plotly
scikit-learn
xgboost
joblib

---

## 🔑 Key Findings

- **Young Adults (20–29)** show the highest average scores across all three conditions
- **Female participants** show higher Depression and Anxiety scores than Male participants
- **Emotional Stability** (TIPI) is the strongest personality predictor of mental health severity
- **Stress, Anxiety, and Depression** are strongly correlated (0.70–0.79) — conditions rarely occur in isolation
- **Stress** has the highest number of High-severity participants

---

## ⚠️ Disclaimer

This dashboard is for **educational purposes only** and is not a clinical diagnosis tool. Results should not replace professional mental health evaluation.

---

## 👤 Author

**Alaa Mekawi**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/alaa-mekawi-37b245221/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/ALAAMEKAWY56)

---

© 2026 · DASS-42 Mental Health Analysis Dashboard

---

## 📁 Project Structure
