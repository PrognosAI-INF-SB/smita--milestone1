# scripts/milestone1_prep.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def get_column_names():
    # FD001 format: unit_number, time_in_cycles, 3 op settings, 21 sensors
    col_names = ["unit_number", "time_in_cycles",
                 "op_setting_1", "op_setting_2", "op_setting_3"]
    col_names += [f"s{i}" for i in range(1, 22)]
    return col_names

def load_cmapps(filepath):
    cols = get_column_names()
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=cols, engine='python')
    return df

def add_rul_column(df):
    max_cycle = df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycle.columns = ["unit_number", "max_cycle"]
    df = df.merge(max_cycle, on="unit_number", how="left")
    df["RUL"] = df["max_cycle"] - df["time_in_cycles"]
    df = df.drop(columns=["max_cycle"])
    return df

def drop_constant_columns(df, feature_cols):
    # drop columns with zero variance
    to_drop = []
    for c in feature_cols:
        if df[c].nunique() <= 1:
            to_drop.append(c)
    if to_drop:
        print(f"Dropping constant columns: {to_drop}")
        feature_cols = [c for c in feature_cols if c not in to_drop]
    return feature_cols

def scale_features(df, feature_cols, scaler_path):
    scaler = StandardScaler()
    df_copy = df.copy()
    df_copy[feature_cols] = scaler.fit_transform(df_copy[feature_cols])
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler -> {scaler_path}")
    return df_copy, scaler

def create_sequences(df, feature_cols, seq_length=30):
    X_list = []
    y_list = []
    unit_ids = []
    for unit in sorted(df["unit_number"].unique()):
        unit_df = df[df["unit_number"] == unit].reset_index(drop=True)
        features = unit_df[feature_cols].values
        rul_vals = unit_df["RUL"].values
        L = len(unit_df)
        if L < seq_length:
            continue
        for start in range(0, L - seq_length + 1):
            seq = features[start:start+seq_length]
            label = rul_vals[start + seq_length - 1]  # RUL at end of sequence
            X_list.append(seq)
            y_list.append(label)
            unit_ids.append(unit)
    X = np.array(X_list)   # shape: (n_samples, seq_length, n_features)
    y = np.array(y_list)   # shape: (n_samples,)
    unit_ids = np.array(unit_ids)
    return X, y, unit_ids

def main(args):
    os.makedirs(args.output_folder, exist_ok=True)

    print("Loading data:", args.input_file)
    df = load_cmapps(args.input_file)
    print("Original shape:", df.shape)

    # Add RUL
    df = add_rul_column(df)
    print("Added RUL column. Sample:")
    print(df.head())

    # feature columns (exclude id/time/RUL)
    feature_cols = ["op_setting_1", "op_setting_2", "op_setting_3"] + [f"s{i}" for i in range(1,22)]

    # Drop constant columns (some sensors might be constant in some datasets)
    feature_cols = drop_constant_columns(df, feature_cols)
    print("Feature columns used:", feature_cols)

    # Scale features and save scaler
    scaler_path = os.path.join(args.output_folder, "scaler_milestone1.joblib")
    df_scaled, scaler = scale_features(df, feature_cols, scaler_path)

    # Save cleaned CSV
    cleaned_csv = os.path.join(args.output_folder, "cleaned_train_FD001.csv")
    df_scaled.to_csv(cleaned_csv, index=False)
    print("Saved cleaned CSV ->", cleaned_csv)

    # Create sequences
    print(f"Creating sequences with seq_length={args.seq_length} ...")
    X, y, unit_ids = create_sequences(df_scaled, feature_cols, seq_length=args.seq_length)
    print("Sequences shapes -- X:", X.shape, "y:", y.shape, "unit_ids:", unit_ids.shape)

    # Save sequences (compressed)
    out_npz = os.path.join(args.output_folder, f"train_sequences_seq{args.seq_length}.npz")
    np.savez_compressed(out_npz, X=X, y=y, unit_ids=unit_ids, feature_cols=np.array(feature_cols))
    print("Saved sequences ->", out_npz)

    # Optionally save as joblib
    joblib.dump({"X": X, "y": y, "unit_ids": unit_ids, "feature_cols": feature_cols},
                os.path.join(args.output_folder, f"train_sequences_seq{args.seq_length}.pkl"))
    print("Also saved joblib .pkl file.")

    print("Milestone 1 preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 1: preprocess CMAPSS FD001")
    parser.add_argument("--input_file", type=str,
                        default=os.path.join("..", "data", "train_FD001.txt"),
                        help="Path to train_FD001.txt")
    parser.add_argument("--seq_length", type=int, default=30, help="Sequence length")
    parser.add_argument("--output_folder", type=str,
                        default=os.path.join("..", "converted_csv"),
                        help="Where to save cleaned data and sequences")
    args = parser.parse_args()
    # adjust path if running from project root
    # if you run from project root: python scripts/milestone1_prep.py --input_file data/train_FD001.txt
    main(args)
