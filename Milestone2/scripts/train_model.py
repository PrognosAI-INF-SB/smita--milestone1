import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "train_FD001_sequences.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Select features and target
feature_cols = [col for col in df.columns if "sensor" in col or "operational_setting" in col]
target_col = "RUL"

X = df[feature_cols].values
y = df[target_col].values

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM [samples, timesteps, features]
# Here timesteps = 1 (simple example)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_scaled, y, epochs=5, batch_size=64, validation_split=0.2, verbose=1)

# Save model
model.save(os.path.join(BASE_DIR, "..", "data", "lstm_model.h5"))
print("✅ Model trained and saved successfully")
