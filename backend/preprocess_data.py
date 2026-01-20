import os
import csv
import numpy as np

DATASET_DIR = "backend/datasets"
GESTURES = ["A", "B", "C", "D"]

X = []
y = []

for label_index, gesture in enumerate(GESTURES):
    csv_path = os.path.join(DATASET_DIR, gesture, f"{gesture}.csv")

    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 63:  # safety check
                X.append([float(val) for val in row])
                y.append(label_index)

X = np.array(X)
y = np.array(y)

# Save processed data
os.makedirs("backend/processed", exist_ok=True)
np.save("backend/processed/X.npy", X)
np.save("backend/processed/y.npy", y)

print("✅ Preprocessing complete")
print("X shape:", X.shape)
print("y shape:", y.shape)
