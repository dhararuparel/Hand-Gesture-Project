import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load the landmark data
df = pd.read_csv("data/image_gestures.csv", header=None)

# Split features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ✅ Debug: Print class distribution
unique, counts = np.unique(y, return_counts=True)
print("Samples per class:", dict(zip(unique, counts)))

# Gesture labels dictionary
gesture_labels = {
    0: 'Closed Fist',
    1: 'OK Sign',
    2: 'Open Palm',
    3: 'Peace Sign',
    4: 'Thumbs Up'
}

# Train on the entire dataset (no test split)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
print("✅ Model trained on entire dataset (no test split)")

# Predict on training data to check
y_pred = clf.predict(X)
print("Classification report on training data:")
print(classification_report(y, y_pred, target_names=[gesture_labels[i] for i in sorted(set(y))], zero_division=0))

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/gesture_classifier.joblib")
print("✅ Model saved as models/gesture_classifier.joblib")

# Save label map
with open("models/gesture_labels.txt", "w") as f:
    for k, v in gesture_labels.items():
        f.write(f"{k},{v}\n")
print("✅ Label map saved to models/gesture_labels.txt")
