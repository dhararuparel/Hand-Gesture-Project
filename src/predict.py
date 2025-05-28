import joblib
import numpy as np

# Load model
clf = joblib.load("models/gesture_classifier.joblib")

# Replace this with a full valid feature vector (same length as in training)
sample_landmarks = np.array([
    0.12, 0.34, 0.56, 0.23, 0.45, 0.67, 0.89, 0.11, 0.33,  # and so on...
    0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.71, 0.82, 0.93,
    0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 0.91,
    0.13, 0.24, 0.35, 0.46, 0.57, 0.68, 0.79, 0.80, 0.81,
    0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.00,
    0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
    0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85
])  # This is 63 values — adjust if yours is different

sample_landmarks = sample_landmarks.reshape(1, -1)

# Predict
pred_class = clf.predict(sample_landmarks)[0]

gesture_labels = {
    0: 'Closed Fist',
    1: 'OK Sign',
    2: 'Open Palm',
    3: 'Peace Sign',
    4: 'Thumbs Up'
}

print("Predicted gesture:", gesture_labels.get(pred_class, "Unknown"))
