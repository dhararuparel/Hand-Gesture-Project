import cv2
import joblib
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
from collections import deque, Counter

# Ensure recordings folder exists
os.makedirs("recordings", exist_ok=True)

# Load your trained gesture classifier
model = joblib.load("models/gesture_classifier.joblib")

# Mapping from label to gesture name
gesture_names = {
    0: 'Closed Fist',
    1: 'OK Sign',
    2: 'Open Palm',
    3: 'Peace Sign',
    4: 'Thumbs Up'
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_features_from_landmarks(hand_landmarks):
    """
    Extract raw x, y, z coordinates from 21 landmarks.
    Match the same format used during training.
    """
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features).reshape(1, -1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"recordings/recording_{timestamp}.avi"

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'XVID'),
                          20.0,
                          (frame_width, frame_height))

    gesture_buffer = deque(maxlen=5)
    stable_gesture = None

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            current_gesture = "No Hand Detected"

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    try:
                        features = extract_features_from_landmarks(hand_landmarks)
                        pred_label = model.predict(features)[0]
                        current_gesture = gesture_names.get(pred_label, "Unknown")
                    except Exception as e:
                        current_gesture = "Prediction Error"
                        print(f"Error: {e}")

                    break  # Only process the first hand

            if current_gesture not in ["No Hand Detected", "Prediction Error"]:
                gesture_buffer.append(current_gesture)
                most_common = Counter(gesture_buffer).most_common(1)[0][0]

                if most_common != stable_gesture:
                    print(most_common)
                    stable_gesture = most_common

                displayed_gesture = stable_gesture
            else:
                displayed_gesture = current_gesture

            # Show gesture on screen
            cv2.putText(frame, f'Gesture: {displayed_gesture}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(frame)
            cv2.imshow('Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
