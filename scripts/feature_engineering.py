import numpy as np

def create_sequences(df, seq_length=30):
    sequences = []
    labels = []
    
    for unit in df["unit_number"].unique():
        unit_data = df[df["unit_number"] == unit]
        unit_data = unit_data.drop(["unit_number", "time_in_cycles"], axis=1).values
        
        for i in range(len(unit_data) - seq_length):
            seq = unit_data[i:i+seq_length]      # past 30 cycles
            label = unit_data[i+seq_length][-1]  # RUL value
            sequences.append(seq)
            labels.append(label)
    
    return np.array(sequences), np.array(labels)

X, y = create_sequences(df, seq_length=30)

print("X shape:", X.shape)  # e.g., (number_of_sequences, 30, features)
print("y shape:", y.shape)


import joblib
import os

output_folder = r"C:\Users\siddharth\OneDrive\Desktop\PrognosAI_Project\converted_csv"
os.makedirs(output_folder, exist_ok=True)

# Save arrays
joblib.dump((X, y), os.path.join(output_folder, "train_sequences.pkl"))
print("Saved preprocessed sequences!")
