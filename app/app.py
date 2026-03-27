import streamlit as st
import numpy as np
import joblib
import os

base = os.path.dirname(__file__)
coef = np.load(os.path.join(base, "../outputs/model_coef.npy"))
intercept = np.load(os.path.join(base, "../outputs/model_intercept.npy"))
scaler = joblib.load(os.path.join(base, "../outputs/scaler.joblib"))

st.title("Insurance Cost Predictor")

age = st.slider("Age", 18, 100)
bmi = st.slider("BMI", 10.0, 50.0)
children = st.slider("Children", 0, 5)
sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox("Region", ["Northwest", "Southeast", "Southwest", "Northeast"])

sex_male = 1 if sex == "Male" else 0
smoker_yes = 1 if smoker == "Yes" else 0
region_northwest = 1 if region == "Northwest" else 0
region_southeast = 1 if region == "Southeast" else 0
region_southwest = 1 if region == "Southwest" else 0

raw = np.array([[age, bmi, children, sex_male, smoker_yes, region_northwest, region_southeast, region_southwest]])
input_scaled = scaler.transform(raw)

if st.button("Predict"):
    prediction = np.dot(input_scaled, coef) + intercept[0]
    st.success(f"Estimated Cost: ${prediction[0]:.2f}")
