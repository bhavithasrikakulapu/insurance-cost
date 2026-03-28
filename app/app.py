import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🏥", layout="wide")

# ── Load models ───────────────────────────────────────────────────────────────
base = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    ridge       = joblib.load(os.path.join(base, "../outputs/ridge_model.joblib"))
    lasso       = joblib.load(os.path.join(base, "../outputs/lasso_model.joblib"))
    poly_model  = joblib.load(os.path.join(base, "../outputs/poly_model.joblib"))
    poly_feat   = joblib.load(os.path.join(base, "../outputs/poly_features.joblib"))
    scaler      = joblib.load(os.path.join(base, "../outputs/scaler.joblib"))
    poly_scaler = joblib.load(os.path.join(base, "../outputs/poly_scaler.joblib"))
    metrics     = joblib.load(os.path.join(base, "../outputs/metrics.joblib"))
    return ridge, lasso, poly_model, poly_feat, scaler, poly_scaler, metrics

ridge, lasso, poly_model, poly_feat, scaler, poly_scaler, metrics = load_models()

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🏥 Insurance Cost Predictor")
st.markdown("Compare **Ridge**, **Lasso**, and **Polynomial Regression** on real insurance data.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "ℹ️ About"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Your Details")

        # BMI Calculator
        with st.expander("🧮 BMI Calculator (optional)"):
            h_col, w_col = st.columns(2)
            with h_col:
                height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            with w_col:
                weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
            calculated_bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
            st.info(f"Your BMI: **{calculated_bmi}**")
            use_calculated = st.checkbox("Use this BMI below")

        age      = st.slider("Age", 18, 100, 30)
        bmi_def  = float(calculated_bmi) if use_calculated else 25.0
        bmi      = st.slider("BMI", 10.0, 50.0, bmi_def)
        children = st.slider("Number of Children", 0, 5, 0)
        sex      = st.selectbox("Sex", ["Male", "Female"])
        smoker   = st.selectbox("Smoker", ["No", "Yes"])
        region   = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

        st.markdown("---")
        model_choice = st.radio(
            "Choose Regression Model",
            ["Ridge", "Lasso", "Polynomial (degree=2)"],
            horizontal=True
        )

    with col2:
        st.subheader("Prediction Result")

        # Encode inputs
        sex_male         = 1 if sex == "Male" else 0
        smoker_yes       = 1 if smoker == "Yes" else 0
        region_northwest = 1 if region == "Northwest" else 0
        region_southeast = 1 if region == "Southeast" else 0
        region_southwest = 1 if region == "Southwest" else 0

        raw = np.array([[age, bmi, children, sex_male, smoker_yes,
                         region_northwest, region_southeast, region_southwest]])

        if st.button("🔮 Predict My Insurance Cost", use_container_width=True):

            X_scaled = scaler.transform(raw)

            if model_choice == "Ridge":
                prediction = ridge.predict(X_scaled)[0]
                m_key = "ridge"
            elif model_choice == "Lasso":
                prediction = lasso.predict(X_scaled)[0]
                m_key = "lasso"
            else:
                X_poly = poly_feat.transform(X_scaled)
                X_poly_scaled = poly_scaler.transform(X_poly)
                prediction = poly_model.predict(X_poly_scaled)[0]
                m_key = "poly"

            prediction = max(prediction, 0)

            # Colour coding
            if prediction < 8000:
                colour, label = "#2ecc71", "Low"
            elif prediction < 20000:
                colour, label = "#f39c12", "Medium"
            else:
                colour, label = "#e74c3c", "High"

            st.markdown(
                f"<h2 style='color:{colour};text-align:center'>${prediction:,.2f} / year</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='text-align:center;font-size:18px'>Risk Level: "
                f"<b style='color:{colour}'>{label}</b> &nbsp;|&nbsp; Model: <b>{model_choice}</b></p>",
                unsafe_allow_html=True,
            )

            # Gauge chart
            fig, ax = plt.subplots(figsize=(5, 1.8))
            max_val = 50000
            ax.barh(0, max_val, color="#ecf0f1", height=0.5)
            ax.barh(0, min(prediction, max_val), color=colour, height=0.5)
            ax.set_xlim(0, max_val)
            ax.set_yticks([])
            ax.set_xlabel("Annual Cost ($)")
            ax.set_title(f"Cost Gauge — {model_choice}")
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig)
            plt.close()

            # Confidence range
            margin = prediction * 0.08
            st.caption(f"Estimated range: ${max(0, prediction - margin):,.0f} – ${prediction + margin:,.0f}")

            # What-if: quitting smoking
            if smoker == "Yes":
                raw_ns = raw.copy()
                raw_ns[0][4] = 0
                X_ns = scaler.transform(raw_ns)
                if model_choice == "Ridge":
                    pred_ns = ridge.predict(X_ns)[0]
                elif model_choice == "Lasso":
                    pred_ns = lasso.predict(X_ns)[0]
                else:
                    pred_ns = poly_model.predict(poly_scaler.transform(poly_feat.transform(X_ns)))[0]
                saving = prediction - pred_ns
                st.info(f"💡 If you quit smoking your cost drops to **${pred_ns:,.2f}** — saving **${saving:,.2f}/year**!")

            # Inline model metrics
            m = metrics[m_key]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("R² Score", m["r2"])
            mc2.metric("MAE",      f"${m['mae']:,.0f}")
            mc3.metric("CV R²",    m["cv"])

            # Save to history
            st.session_state.history.append({
                "Model": model_choice, "Age": age, "BMI": round(bmi, 1),
                "Smoker": smoker, "Region": region,
                "Prediction": f"${prediction:,.2f}"
            })

        # History table
        if st.session_state.history:
            st.markdown("---")
            st.subheader("📋 Last 5 Predictions")
            df_hist = pd.DataFrame(st.session_state.history[-5:])
            st.dataframe(df_hist, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Model Comparison")

    perf_data = {
        "Model":    ["Ridge Regression", "Lasso Regression", "Polynomial (degree=2)"],
        "R² Score": [metrics["ridge"]["r2"], metrics["lasso"]["r2"], metrics["poly"]["r2"]],
        "MAE ($)":  [metrics["ridge"]["mae"], metrics["lasso"]["mae"], metrics["poly"]["mae"]],
        "CV R²":    [metrics["ridge"]["cv"],  metrics["lasso"]["cv"],  metrics["poly"]["cv"]],
    }
    df_perf = pd.DataFrame(perf_data)
    st.dataframe(
        df_perf.style.highlight_max(subset=["R² Score", "CV R²"], color="lightgreen")
                     .highlight_min(subset=["MAE ($)"],            color="lightgreen"),
        use_container_width=True
    )

    st.markdown("---")

    # Bar charts
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    models = ["Ridge", "Lasso", "Polynomial"]
    r2s    = [metrics["ridge"]["r2"], metrics["lasso"]["r2"], metrics["poly"]["r2"]]
    maes   = [metrics["ridge"]["mae"], metrics["lasso"]["mae"], metrics["poly"]["mae"]]
    colors = ["#3498db", "#9b59b6", "#e67e22"]

    axes[0].bar(models, r2s, color=colors)
    axes[0].set_title("R² Score (higher is better)")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("R²")
    for i, v in enumerate(r2s):
        axes[0].text(i, v + 0.01, str(v), ha="center", fontsize=10, fontweight="bold")

    axes[1].bar(models, maes, color=colors)
    axes[1].set_title("MAE — $ Error (lower is better)")
    axes[1].set_ylabel("MAE ($)")
    for i, v in enumerate(maes):
        axes[1].text(i, v + 50, f"${v:,.0f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Residual plot
    st.subheader("Residual Plot — Polynomial Model")
    st.caption("Good models have residuals scattered randomly around zero.")

    df_raw = pd.read_csv(os.path.join(base, "../data/insurance.csv"))
    df_raw = pd.get_dummies(df_raw, drop_first=True)
    X_all  = df_raw.drop("charges", axis=1).values
    y_all  = df_raw["charges"].values

    X_all_scaled  = scaler.transform(X_all)
    X_all_poly    = poly_feat.transform(X_all_scaled)
    X_all_poly_s  = poly_scaler.transform(X_all_poly)
    y_pred_all    = poly_model.predict(X_all_poly_s)
    residuals     = y_all - y_pred_all

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(y_pred_all, residuals, alpha=0.4, color="#e67e22", s=20)
    ax2.axhline(0, color="black", linewidth=1, linestyle="--")
    ax2.set_xlabel("Predicted Values ($)")
    ax2.set_ylabel("Residuals ($)")
    ax2.set_title("Residuals vs Predicted — Polynomial Regression")
    st.pyplot(fig2)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ About This App")
    st.markdown("""
    This app predicts annual health insurance costs and compares three linear regression approaches.

    **Models**
    - **Ridge Regression** — L2 regularization, penalizes large coefficients
    - **Lasso Regression** — L1 regularization, performs automatic feature selection
    - **Polynomial Regression** — adds interaction terms to capture non-linear patterns

    **Tech Stack**
    - Python, Scikit-learn, NumPy, Pandas, Matplotlib
    - Streamlit for deployment
    - Custom Gradient Descent implementation (src/gd.py)

    **Dataset**
    - 1,338 insurance records
    - Features: Age, BMI, Children, Sex, Smoker status, Region

    **Key Insight**
    Smoking status is the single biggest driver of insurance cost —
    smokers pay on average **3–4× more** than non-smokers.
    Polynomial Regression captures this non-linear relationship best among linear models.
    """)
