# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"data/after_edat_credit_approval_clean.csv")

df_encoded = pd.get_dummies(df, drop_first=True)

# Step 2: Scale the features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

k = 2

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

label = kmeans.predict(X_scaled)
print(label)

df["approved"] = label

df.to_csv("data/label_data.csv")

