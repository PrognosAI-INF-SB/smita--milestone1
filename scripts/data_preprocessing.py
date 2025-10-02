import pandas as pd
import os

# File path
data_path = r"C:\Users\siddharth\OneDrive\Desktop\PrognosAI_Project\data\train_FD001.txt"

# Column names (NASA CMAPSS FD001: 26 cols → ID, cycle, 3 settings, 21 sensors)
col_names = ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"] \
             + [f"s{i}" for i in range(1,22)]

# Load dataset
df = pd.read_csv(data_path, sep="\s+", header=None, names=col_names)

print("Data shape:", df.shape)
print(df.head())


def add_rul(df):
    # Find max cycle for each engine
    max_cycle = df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycle.columns = ["unit_number", "max_cycle"]
    
    # Merge with dataset
    df = df.merge(max_cycle, on="unit_number", how="left")
    
    # Calculate RUL
    df["RUL"] = df["max_cycle"] - df["time_in_cycles"]
    return df.drop("max_cycle", axis=1)

df = add_rul(df)
print(df.head())
