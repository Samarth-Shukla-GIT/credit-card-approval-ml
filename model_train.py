# feature_eng_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/label_data.csv")  # replace with your CSV file name

# Drop index column if it exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Split features and target
X = df.drop(["approved", "ID"], axis=1)
y = df["approved"]

# 🔹 Drop unwanted columns
drop_cols = ["DAYS_BIRTH", "DAYS_EMPLOYED"]
X = X.drop(columns=[c for c in drop_cols if c in X.columns])

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocessing
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numerical_transformer, numerical_cols)
    ]
)

# Pipeline with Logistic Regression
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Save the full pipeline (encoder + model)
with open("credit_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and preprocessing pipeline saved as credit_model.pkl")

