import streamlit as st
import joblib
import numpy as np

model = joblib.load("outputs/model.joblib")

st.title("Insurance Cost Predictor")

age = st.slider("Age", 18, 100)
bmi = st.slider("BMI", 10.0, 50.0)
children = st.slider("Children", 0, 5)
smoker = st.selectbox("Smoker", ["Yes", "No"])

# NOTE: This must match training features order
smoker_val = 1 if smoker == "Yes" else 0

# Minimal feature vector (adjust if needed)
input_data = np.array([[age, bmi, children, smoker_val]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Cost: ${prediction:.2f}")

