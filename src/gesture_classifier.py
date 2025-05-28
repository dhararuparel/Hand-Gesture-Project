import joblib
import numpy as np
from feature_extraction import landmarks_to_features

class GestureClassifier:
    def __init__(self, model_path='models/gesture_classifier.joblib'):
        print("✅ Loading gesture classifier model...")
        self.model = joblib.load(model_path)

        # Class label mapping — must match training order
        self.gesture_labels = {
            0: 'Closed Fist',
            1: 'OK Sign',
            2: 'Open Palm',
            3: 'Peace Sign',
            4: 'Thumbs Up'
        }

    def normalize_landmarks(self, landmarks):
        """
        Normalize landmark coordinates by subtracting the wrist and scaling to unit length.
        This must match the preprocessing used during training.
        """
        coords = np.array(landmarks).reshape(-1, 3)
        base = coords[0]  # wrist (landmark 0)
        coords -= base  # translation to origin

        max_value = np.max(np.linalg.norm(coords, axis=1))
        if max_value != 0:
            coords /= max_value  # scale to unit length

        return coords.flatten().tolist()

    def predict_from_csv_row(self, flat_landmarks):
        """
        Predict from a list of 63 landmark values (x, y, z for 21 points).
        """
        features = landmarks_to_features(flat_landmarks).reshape(1, -1)
        proba = self.model.predict_proba(features)
        pred_class = np.argmax(proba)
        print(f"🔍 Prediction Probabilities: {proba}")
        print(f"🧠 Predicted Class Index: {pred_class}")
        return self.gesture_labels.get(pred_class, "Unknown")

    def predict(self, hand_landmarks):
        """
        Predict gesture from MediaPipe hand landmarks object.
        """
        flat_landmarks = []
        for lm in hand_landmarks.landmark:
            flat_landmarks.extend([lm.x, lm.y, lm.z])

        # Normalize landmarks (must match training)
        flat_landmarks = self.normalize_landmarks(flat_landmarks)

        return self.predict_from_csv_row(flat_landmarks)
