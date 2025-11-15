import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# STEP 1 — Load Model & Scaler
# -----------------------------
knn_model = joblib.load("knn_walmart_model.pkl")
scaler = joblib.load("scaler_walmart.pkl")

# -----------------------------
# STEP 2 — App Layout
# -----------------------------

# Sidebar: Project Info
st.sidebar.header("About This Project")
st.sidebar.info("""
This app predicts **Weekly Sales** for Walmart stores using a **KNN Regression model**.  
You can adjust the input features in the main page and get real-time predictions.  
- Model trained on historical Walmart dataset  
- Features include Store info, Holiday flag, Temperature, Fuel Price, CPI, Unemployment, Year, Month, Week
""")

# Main Page: Title
st.title("Walmart Weekly Sales Prediction")

# -----------------------------
# STEP 3 — Feature Inputs
# -----------------------------
st.subheader("Input Features")

# Define features and default values
feature_defaults = {
    'Store': 1,
    'Holiday_Flag': 0,
    'Temperature': 60.0,
    'Fuel_Price': 3.5,
    'CPI': 180.0,
    'Unemployment': 8.0,
    'Year': 2012,
    'Month': 1,
    'Week': 1
}

# Store user inputs in dictionary
user_inputs = {}

# Create input widgets in main page
for feature, default in feature_defaults.items():
    if feature in ['Store', 'Holiday_Flag', 'Year', 'Month', 'Week']:
        # Integer inputs
        user_inputs[feature] = st.number_input(
            label=feature,
            min_value=0,
            max_value=100,
            value=default,
            step=1
        )
    else:
        # Float inputs
        user_inputs[feature] = st.number_input(
            label=feature,
            min_value=0.0,
            max_value=10000.0,
            value=float(default),
            step=0.01,
            format="%.2f"
        )

# Convert inputs to DataFrame
input_df = pd.DataFrame(user_inputs, index=[0])

# -----------------------------
# STEP 4 — Scale Inputs & Predict
# -----------------------------
input_scaled = scaler.transform(input_df)
prediction = knn_model.predict(input_scaled)[0]

# -----------------------------
# STEP 5 — Display Prediction
# -----------------------------
st.subheader("Predicted Weekly Sales")
st.success(f"${prediction:,.2f}")