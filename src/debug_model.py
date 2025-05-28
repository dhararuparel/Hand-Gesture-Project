import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def debug_dataset():
    """Debug the training dataset"""
    print("🔍 Debugging dataset...")
    
    if not os.path.exists("data/image_gestures.csv"):
        print("❌ Dataset not found! Please run extract_features_from_images.py first")
        return False
    
    # Load dataset
    df = pd.read_csv("data/image_gestures.csv", header=None)
    print(f"📊 Dataset shape: {df.shape}")
    
    # Check labels
    labels = df.iloc[:, -1].values
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print(f"📋 Unique labels: {unique_labels}")
    print(f"📊 Label counts: {counts}")
    
    gesture_names = {0: 'Closed Fist', 1: 'OK Sign', 2: 'Open Palm', 3: 'Peace Sign', 4: 'Thumbs Up'}
    
    print("\n📈 Class distribution:")
    for label, count in zip(unique_labels, counts):
        gesture_name = gesture_names.get(label, f"Unknown({label})")
        print(f"  {gesture_name}: {count} samples")
    
    # Check for class imbalance
    if len(set(counts)) > 1:
        print("⚠️ WARNING: Class imbalance detected!")
        if min(counts) < 5:
            print("❌ Some classes have very few samples - this will cause poor recognition!")
    
    # Check feature consistency
    features = df.iloc[:, :-1].values
    print(f"🔢 Feature shape: {features.shape}")
    print(f"📊 Feature range: {features.min():.3f} to {features.max():.3f}")
    
    return True

def retrain_improved_model():
    """Retrain model with better parameters"""
    print("\n🎯 Retraining model with improved settings...")
    
    # Load data
    df = pd.read_csv("data/image_gestures.csv", header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    if len(X) < 10:
        print("⚠️ Very small dataset - training on all data")
        X_train, X_test = X, X
        y_train, y_test = y, y
    else:
        test_size = int(0.2 * len(X))
        unique_classes = len(np.unique(y))
        
        if test_size < unique_classes:
            print("⚠️ Test size too small for stratified split, splitting without stratify")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=None
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
    
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Test samples: {len(X_test)}")
    
    # Train with better parameters
    clf = RandomForestClassifier(
        n_estimators=200,           # More trees
        max_depth=10,               # Prevent overfitting
        min_samples_split=2,        # Minimum samples to split
        min_samples_leaf=1,         # Minimum samples in leaf
        random_state=42,
        class_weight='balanced'     # Handle class imbalance
    )
    
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    
    gesture_names = {0: 'Closed Fist', 1: 'OK Sign', 2: 'Open Palm', 3: 'Peace Sign', 4: 'Thumbs Up'}
    target_names = [gesture_names[i] for i in sorted(set(y))]
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    # Feature importance
    feature_importance = clf.feature_importances_
    print(f"\n🔍 Top 10 most important features:")
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(top_indices):
        landmark_idx = idx // 3
        coord = ['x', 'y', 'z'][idx % 3]
        print(f"  {i+1}. Landmark {landmark_idx} ({coord}): {feature_importance[idx]:.4f}")
    
    # Save improved model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/gesture_classifier.joblib")
    print("✅ Improved model saved!")
    
    return clf

def test_model_predictions():
    """Test model with some sample predictions"""
    print("\n🧪 Testing model predictions...")
    
    try:
        clf = joblib.load("models/gesture_classifier.joblib")
        
        # Load some test data
        df = pd.read_csv("data/image_gestures.csv", header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        gesture_names = {0: 'Closed Fist', 1: 'OK Sign', 2: 'Open Palm', 3: 'Peace Sign', 4: 'Thumbs Up'}
        
        # Test on a few samples
        print("\n🎯 Sample predictions:")
        for i in range(min(10, len(X))):
            sample = X[i:i+1]
            true_label = y[i]
            
            # Get prediction probabilities
            proba = clf.predict_proba(sample)[0]
            pred_label = np.argmax(proba)
            confidence = proba[pred_label]
            
            true_gesture = gesture_names.get(true_label, "Unknown")
            pred_gesture = gesture_names.get(pred_label, "Unknown")
            
            status = "✅" if pred_label == true_label else "❌"
            print(f"  {status} True: {true_gesture}, Predicted: {pred_gesture} (conf: {confidence:.3f})")
            
            # Show all probabilities
            print(f"    Probabilities: ", end="")
            for j, prob in enumerate(proba):
                gesture = gesture_names.get(j, f"Class{j}")
                print(f"{gesture}: {prob:.3f} ", end="")
            print()
    
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def create_more_training_data():
    """Suggest how to create more training data"""
    print("\n💡 Tips to improve your model:")
    print("1. 📸 Add more images for each gesture (aim for 20+ per gesture)")
    print("2. 🔄 Use different hand positions, angles, and lighting")
    print("3. 👥 Include images from different people")
    print("4. 🖐️ Make sure gestures are clearly distinct")
    print("5. 🎯 Check that your gesture images match the names exactly")
    
    print("\n📁 Expected image names in data/images/:")
    expected_files = [
        "closed_fist.jpg/png", "ok_sign.jpg/png", "open_palm.jpg/png", 
        "peace_sign.jpg/png", "thumbs_up.jpg/png"
    ]
    for file in expected_files:
        print(f"  - {file}")

def main():
    print("🔧 Gesture Recognition Model Debugger")
    print("="*50)
    
    # Debug dataset
    if not debug_dataset():
        return
    
    # Retrain model
    retrain_improved_model()
    
    # Test predictions
    test_model_predictions()
    
    # Suggestions
    create_more_training_data()
    
    print("\n🎉 Debugging complete! Try running your gesture recognition again.")

if __name__ == "__main__":
    main()
