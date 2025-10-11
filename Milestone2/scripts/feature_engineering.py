import pandas as pd
import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "train_FD001_clean.csv")

# Load preprocessed CSV
df = pd.read_csv(DATA_PATH)

# Compute Remaining Useful Life (RUL)
rul = df.groupby("unit_number")["time_in_cycles"].max().reset_index()
rul.columns = ["unit_number", "max_cycles"]
df = df.merge(rul, on="unit_number", how="left")
df["RUL"] = df["max_cycles"] - df["time_in_cycles"]

# Drop max_cycles column
df.drop("max_cycles", axis=1, inplace=True)

print("✅ Feature engineering done")
print(df.head())

# Save sequences for training
df.to_csv(os.path.join(BASE_DIR, "..", "data", "train_FD001_sequences.csv"), index=False)
