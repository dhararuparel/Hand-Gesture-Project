import os
import cv2
import numpy as np
import mediapipe as mp
import csv

# Paths
IMAGE_DIR = "data/images"
OUTPUT_CSV = "data/image_gestures.csv"

# Gesture label mapping
gesture_map = {
    "closed_fist": 0,
    "ok_sign": 1,
    "open_palm": 2,
    "peace_sign": 3,
    "thumbs_up": 4
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Create output CSV file
os.makedirs("data", exist_ok=True)
with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    
    # Process each image
    for filename in os.listdir(IMAGE_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Detect label from filename
        for gesture_name, label in gesture_map.items():
            if filename.startswith(gesture_name):
                image_path = os.path.join(IMAGE_DIR, filename)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"❌ Could not load {filename}")
                    continue
                
                # Convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = hands.process(image_rgb)
                
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Extract 3D landmarks (x, y, z)
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        
                        # Append label
                        landmarks.append(label)
                        writer.writerow(landmarks)
                        print(f"✅ Processed: {filename}")
                else:
                    print(f"⚠️ No hand detected in: {filename}")
                break  # Once a label is matched, skip checking others

print(f"\n📁 Landmark data saved to: {OUTPUT_CSV}")
