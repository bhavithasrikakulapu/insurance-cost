import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("charges", axis=1)
    y = df["charges"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, X.columns, scaler
