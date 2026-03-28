# 🏥 Insurance Cost Predictor

> Comparing Ridge, Lasso, and Polynomial Regression on real US medical insurance data — deployed as an interactive web app.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://insurance-cost-lbbdutyyablvzabpt69ahx.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/bhavithasrikakulapu/insurance-cost)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)

---

## 📌 Overview

This project explores how different linear regression approaches perform on predicting annual US health insurance costs. The focus is on understanding **why models behave differently** — not just building one and stopping there.

Built end-to-end: from custom gradient descent implementation to a live deployed web application.

---

## 🔬 Models Compared

| Model | R² Score | MAE | CV R² |
|---|---|---|---|
| Ridge Regression | 0.7509 | $4,172 | 0.7469 |
| Lasso Regression | 0.7509 | $4,171 | 0.7469 |
| **Polynomial Regression (degree=2)** | **0.8478** | **$2,836** | **0.8354** |

Polynomial Regression outperforms standard linear models by capturing non-linear interactions between features — particularly the relationship between smoking status and cost.

---

## ✨ App Features

- **Live model switching** — compare Ridge, Lasso and Polynomial predictions in real time
- **BMI Calculator** — auto-calculate BMI from height and weight
- **What-if Analysis** — see how quitting smoking reduces your estimated cost
- **Cost Gauge** — visual indicator with color coded risk levels (Low / Medium / High)
- **Confidence Range** — estimated cost range for each prediction
- **Prediction History** — track and compare last 5 predictions
- **Model Performance Tab** — R², MAE, CV R² comparison with bar charts and residual plot

---

## 📁 Project Structure

```
insurance-cost/
├── app/
│   └── app.py              # Streamlit web application
├── src/
│   ├── train.py            # Model training script
│   ├── preprocess.py       # Data loading and preprocessing
│   ├── gd.py               # Gradient descent from scratch
│   └── visualize.py        # Learning curve visualizations
├── data/
│   └── insurance.csv       # Medical Cost Personal Dataset (Kaggle)
├── outputs/
│   ├── ridge_model.joblib
│   ├── lasso_model.joblib
│   ├── poly_model.joblib
│   ├── poly_features.joblib
│   ├── scaler.joblib
│   ├── poly_scaler.joblib
│   └── metrics.joblib
├── requirements.txt
└── runtime.txt
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML & Data | Scikit-learn, NumPy, Pandas |
| Visualization | Matplotlib |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/bhavithasrikakulapu/insurance-cost.git
cd insurance-cost

# Install dependencies
pip install -r requirements.txt

# Train models
python src/train.py

# Launch app
streamlit run app/app.py
```

---

## 💡 Key Insight

Smoking status is the single biggest cost driver in this dataset — smokers pay **3–4× more** than non-smokers on average. Standard linear models struggle to capture this non-linearity. Adding polynomial interaction terms bridges that gap, improving R² from **0.75 → 0.85**.

---

## 📊 Dataset

- **Source:** [Medical Cost Personal Dataset — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Records:** 1,338
- **Features:** Age, BMI, Children, Sex, Smoker status, Region
- **Target:** Annual insurance charges (USD)
- **Note:** Dataset reflects US private healthcare pricing.

---

## 🔗 Live Demo

👉 [Try the app here](https://insurance-cost-lbbdutyyablvzabpt69ahx.streamlit.app/)
