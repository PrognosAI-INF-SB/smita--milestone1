import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "train_FD001_sequences.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "data", "lstm_model.h5")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Features & target
feature_cols = [col for col in df.columns if "sensor" in col or "operational_setting" in col]
target_col = "RUL"

X = df[feature_cols].values
y_actual = df[target_col].values

# Load trained model
from tensorflow.keras.losses import MeanSquaredError

model = load_model(MODEL_PATH, compile=False)  # don't compile automatically
model.compile(optimizer='adam', loss=MeanSquaredError())


# Scale features (same as training)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Make predictions
y_pred = model.predict(X_scaled, verbose=0).flatten()

# Plot actual vs predicted RUL
plt.figure(figsize=(12,6))
plt.plot(y_actual, label="Actual RUL", color="blue")
plt.plot(y_pred, label="Predicted RUL", color="red", alpha=0.7)
plt.title("Remaining Useful Life (RUL) Prediction")
plt.xlabel("Samples")
plt.ylabel("RUL")
plt.legend()
plt.show()

# Optional: Save plot
plt.savefig(os.path.join(BASE_DIR, "..", "data", "RUL_predictions_plot.png"))
print("✅ Visualization complete and plot saved")
