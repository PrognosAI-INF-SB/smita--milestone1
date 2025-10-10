import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# Load preprocessed CSV
DATA_FILE = "../data/converted_csv/train_FD001_processed.csv"
df = pd.read_csv(DATA_FILE)

# Only use sensor columns for input
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
X = df[sensor_cols].values
y = df["RUL"].values

# Normalize X
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences for LSTM
def create_sequences(X, y, seq_length=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

SEQ_LENGTH = 30
X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LENGTH)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), activation="tanh"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.summary()

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Save trained model
os.makedirs("../models", exist_ok=True)
model.save("../models/lstm_rul_model.h5")
print("Model trained and saved to ../models/lstm_rul_model.h5")
