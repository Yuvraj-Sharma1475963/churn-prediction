from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

MODEL_PATH = Path(__file__).parent / "churn_model.joblib"


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


artifacts = load_artifacts()
model = artifacts["model"]
columns = artifacts["columns"]
threshold = artifacts["threshold"]
options = artifacts["options"]
explainer = load_explainer(model)

INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]

RAW_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] + list(options)


def preprocess(raw_df, columns):
    if set(raw_df.columns) != set(RAW_COLUMNS):
        raise ValueError(f"Input columns don't match training: {sorted(set(raw_df.columns) ^ set(RAW_COLUMNS))}")
    encoded = pd.get_dummies(raw_df)
    return encoded.reindex(columns=columns, fill_value=False)


st.title("Customer Churn Risk")
st.write(f"Model loaded: {len(columns)} features, threshold {threshold}")

with st.form("customer_form"):
    st.subheader("Customer details")

    senior = st.radio("Senior citizen?", ["No", "Yes"], horizontal=True)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    monthly = st.number_input("Monthly charges", min_value=0.0, max_value=200.0, value=70.0)
    total = st.number_input("Total charges", min_value=0.0, max_value=10000.0, value=840.0)

    inputs = {
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    grid = st.columns(3)
    for i, (col, values) in enumerate(options.items()):
        with grid[i % 3]:
            inputs[col] = st.selectbox(col, values)

    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    # 1. Auto-fix fields that have only one valid value, and record what changed
    notes = []
    if inputs["PhoneService"] == "No" and inputs["MultipleLines"] != "No phone service":
        inputs["MultipleLines"] = "No phone service"
        notes.append("MultipleLines was set to 'No phone service' because the customer has no phone service.")
    if inputs["InternetService"] == "No":
        changed = [c for c in INTERNET_ADDONS if inputs[c] != "No internet service"]
        for c in INTERNET_ADDONS:
            inputs[c] = "No internet service"
        if changed:
            notes.append(f"Set to 'No internet service' because the customer has no internet: {', '.join(changed)}.")

    # 2. Block contradictions and combinations the model never saw in training
    problems = []
    if inputs["PhoneService"] == "No" and inputs["InternetService"] != "DSL":
        problems.append(
            "No training data for customers without phone service unless they have DSL internet "
            f"(all 682 such customers have DSL; you chose '{inputs['InternetService']}'), "
            "so the model can't score this reliably."
        )
    if inputs["PhoneService"] == "Yes" and inputs["MultipleLines"] == "No phone service":
        problems.append("MultipleLines can't be 'No phone service' when PhoneService is Yes.")
    for c in INTERNET_ADDONS:
        if inputs["InternetService"] != "No" and inputs[c] == "No internet service":
            problems.append(f"{c} can't be 'No internet service' when the customer has internet.")

    if problems:
        for p in problems:
            st.error(p)
        st.stop()

    for n in notes:
        st.info(n)

    # 3. Predict
    row = preprocess(pd.DataFrame([inputs]), columns)
    score = model.predict_proba(row)[0, 1]

    st.metric("Churn risk score", f"{score:.2f}")
    if score >= threshold:
        st.error(f"HIGH RISK: score is at or above the {threshold:.2f} threshold.")
    else:
        st.success(f"LOW RISK: score is below the {threshold:.2f} threshold.")
    st.caption("This score ranks customers by risk. It is not a calibrated probability.")

    # 4. Explain the score
    sv = explainer(row)
    reasons = pd.Series(sv.values[0], index=columns).sort_values(ascending=False)

    st.subheader("Why this score?")
    st.write("Biggest factors raising this customer's risk:")
    for name, value in reasons.head(3).items():
        if value > 0:
            st.write(f"- **{name}** = {row[name].iloc[0]}  (+{value:.2f})")

    shap.plots.waterfall(sv[0], show=False)
    st.pyplot(plt.gcf())
    plt.close()