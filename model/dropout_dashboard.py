import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="🎓 Student Dropout Prediction", layout="wide")

st.title("🎓 Student Dropout Prediction Dashboard")

# Load model and preprocessors
model_path = "../model/rf_model.pkl"
scaler_path = "../model/scaler.pkl"
feature_path = "../model/feature_names.pkl"

if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_path)):
    st.error("❌ Model not trained yet. Please run train_model.py first.")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(feature_path)

# File uploader
uploaded_file = st.file_uploader("📂 Upload Student Data CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Preprocess
    X = data[feature_names].fillna(data.mean())
    X_scaled = scaler.transform(X)

    # Predictions
    probs = model.predict_proba(X_scaled)[:, 1] * 100
    preds = ["High Risk" if p >= 70 else "Medium Risk" if p >= 30 else "Low Risk" for p in probs]

    data["Dropout Risk (%)"] = probs
    data["Category"] = preds

    st.subheader("📋 Predictions")
    st.dataframe(data)

    # Download option
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results", data=csv, file_name="predictions.csv", mime="text/csv")

    # Risk distribution
    fig = px.histogram(data, x="Category", title="📊 Risk Distribution", color="Category")
    st.plotly_chart(fig, use_container_width=True)
