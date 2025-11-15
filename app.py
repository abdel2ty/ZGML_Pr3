import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# -----------------------------
# STEP 1 — Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Walmart.csv")  # ضع ملفك في نفس فولدر app.py
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    return df

df = load_data()

# -----------------------------
# STEP 2 — Prepare Features & Target
# -----------------------------
features = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
            'CPI', 'Unemployment', 'Year', 'Month', 'Week']
target = 'Weekly_Sales'

X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# STEP 3 — Train Tuned KNN
# -----------------------------
@st.cache_resource
def train_model(X_scaled, y):
    knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn.fit(X_scaled, y)
    return knn

knn_model = train_model(X_scaled, y)

# -----------------------------
# STEP 4 — Streamlit Layout
# -----------------------------
# Sidebar: Project Info
st.sidebar.header("About This Project")
st.sidebar.info("""
This app predicts **Weekly Sales** for Walmart stores using a **KNN Regression model**.  
Adjust the input features on the main page and get real-time predictions.  

- Model trained on historical Walmart dataset  
- Features: Store, Holiday flag, Temperature, Fuel Price, CPI, Unemployment, Year, Month, Week
""")

# Main Page: Title
st.title("Walmart Weekly Sales Prediction")

# Feature Inputs
st.subheader("Input Features")

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

user_inputs = {}

for feature, default in feature_defaults.items():
    if feature == 'Store':
        user_inputs[feature] = st.number_input(feature, min_value=1, max_value=50, value=default, step=1)
    elif feature == 'Holiday_Flag':
        user_inputs[feature] = st.number_input(feature, min_value=0, max_value=1, value=default, step=1)
    elif feature == 'Year':
        user_inputs[feature] = st.number_input(feature, min_value=2010, max_value=2025, value=default, step=1)
    elif feature == 'Month':
        user_inputs[feature] = st.number_input(feature, min_value=1, max_value=12, value=default, step=1)
    elif feature == 'Week':
        user_inputs[feature] = st.number_input(feature, min_value=1, max_value=53, value=default, step=1)
    else:  # Float inputs
        # اختيار max_value كبير يكفي لتغطية الداتا
        user_inputs[feature] = st.number_input(feature, min_value=0.0, max_value=1000.0, value=float(default), step=0.01, format="%.2f")

input_df = pd.DataFrame(user_inputs, index=[0])

# -----------------------------
# STEP 5 — Scale & Predict
# -----------------------------
input_scaled = scaler.transform(input_df)
prediction = knn_model.predict(input_scaled)[0]

# Display prediction
st.subheader("Predicted Weekly Sales")
st.success(f"${prediction:,.2f}")

# Optional: Show dataset sample
if st.checkbox("Show Dataset Sample"):
    st.dataframe(df.head(10))