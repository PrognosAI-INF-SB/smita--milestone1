import pandas as pd
import os

# Set dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "train_FD001.txt")

# Column names for FD001 dataset
col_names = [
    "unit_number", "time_in_cycles",
    "operational_setting_1", "operational_setting_2", "operational_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"
]

# Read dataset using space separator
df = pd.read_csv(DATA_PATH, sep="\s+", header=None, names=col_names)

print("✅ Dataset loaded successfully")
print("Shape:", df.shape)
print(df.head())

# Save a clean version as CSV for next steps (optional)
df.to_csv(os.path.join(BASE_DIR, "..", "data", "train_FD001_clean.csv"), index=False)
