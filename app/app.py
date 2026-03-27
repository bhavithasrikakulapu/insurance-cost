import joblib
import streamlit as st
import numpy as np
import os

coef = np.load(os.path.join(os.path.dirname(__file__), "../outputs/model_coef.npy"))
intercept = np.load(os.path.join(os.path.dirname(__file__), "../outputs/model_intercept.npy"))

st.title("Insurance Cost Predictor")

age = st.slider("Age", 18, 100)
bmi = st.slider("BMI", 10.0, 50.0)
children = st.slider("Children", 0, 5)
smoker = st.selectbox("Smoker", ["Yes", "No"])
sex = st.selectbox("Sex", ["Male", "Female"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

smoker_val = 1 if smoker == "Yes" else 0
sex_val = 1 if sex == "Male" else 0
region_northeast = 1 if region == "Northeast" else 0
region_northwest = 1 if region == "Northwest" else 0
region_southeast = 1 if region == "Southeast" else 0

input_data = np.array([age, bmi, children, smoker_val, sex_val, region_northeast, region_northwest, region_southeast])

if st.button("Predict"):
    prediction = np.dot(input_data, coef) + intercept[0]
    st.success(f"Estimated Cost: ${prediction:.2f}")

