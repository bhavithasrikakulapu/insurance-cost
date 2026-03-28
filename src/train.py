import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib
from preprocess import load_and_preprocess
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

# ── Load & preprocess ────────────────────────────────────────────────────────
# Run from repo root: python src/train.py
X_scaled, y, feature_names, scaler = load_and_preprocess("data/insurance.csv")

# ── Ridge Regression ─────────────────────────────────────────────────────────
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
ridge_r2  = r2_score(y, ridge.predict(X_scaled))
ridge_mae = mean_absolute_error(y, ridge.predict(X_scaled))
ridge_cv  = cross_val_score(ridge, X_scaled, y, cv=5, scoring="r2").mean()
print(f"Ridge      → R²: {ridge_r2:.4f} | MAE: ${ridge_mae:.2f} | CV R²: {ridge_cv:.4f}")

# ── Lasso Regression ─────────────────────────────────────────────────────────
lasso = Lasso(alpha=1.0)
lasso.fit(X_scaled, y)
lasso_r2  = r2_score(y, lasso.predict(X_scaled))
lasso_mae = mean_absolute_error(y, lasso.predict(X_scaled))
lasso_cv  = cross_val_score(lasso, X_scaled, y, cv=5, scoring="r2").mean()
print(f"Lasso      → R²: {lasso_r2:.4f} | MAE: ${lasso_mae:.2f} | CV R²: {lasso_cv:.4f}")

# ── Polynomial Regression (degree=2) ─────────────────────────────────────────
poly_feat = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_feat.fit_transform(X_scaled)
poly_scaler = StandardScaler()
X_poly_scaled = poly_scaler.fit_transform(X_poly)

poly_model = Ridge(alpha=1.0)
poly_model.fit(X_poly_scaled, y)
poly_r2  = r2_score(y, poly_model.predict(X_poly_scaled))
poly_mae = mean_absolute_error(y, poly_model.predict(X_poly_scaled))
poly_cv  = cross_val_score(poly_model, X_poly_scaled, y, cv=5, scoring="r2").mean()
print(f"Polynomial → R²: {poly_r2:.4f} | MAE: ${poly_mae:.2f} | CV R²: {poly_cv:.4f}")

# ── Save all models & scalers ─────────────────────────────────────────────────
joblib.dump(ridge,       "outputs/ridge_model.joblib",   protocol=2)
joblib.dump(lasso,       "outputs/lasso_model.joblib",   protocol=2)
joblib.dump(poly_model,  "outputs/poly_model.joblib",    protocol=2)
joblib.dump(poly_feat,   "outputs/poly_features.joblib", protocol=2)
joblib.dump(scaler,      "outputs/scaler.joblib",        protocol=2)
joblib.dump(poly_scaler, "outputs/poly_scaler.joblib",   protocol=2)

# Save metrics for app
metrics = {
    "ridge": {"r2": round(ridge_r2, 4), "mae": round(ridge_mae, 2), "cv": round(ridge_cv, 4)},
    "lasso": {"r2": round(lasso_r2, 4), "mae": round(lasso_mae, 2), "cv": round(lasso_cv, 4)},
    "poly":  {"r2": round(poly_r2,  4), "mae": round(poly_mae,  2), "cv": round(poly_cv,  4)},
}
joblib.dump(metrics, "outputs/metrics.joblib", protocol=2)

print("\nAll models saved to outputs/ successfully!")
