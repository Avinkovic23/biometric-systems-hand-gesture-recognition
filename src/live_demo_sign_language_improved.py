from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import subprocess
import time
import pygetwindow as gw

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "sign_language_model_current_data.joblib"

if not MODEL_PATH.exists():
    print("Improved model not found, using original model...")
    MODEL_PATH = MODELS_DIR / "sign_language_model.joblib"
    USE_IMPROVED_MODEL = False
else:
    print("Using improved model with feature engineering")
    USE_IMPROVED_MODEL = True

letters = [chr(ord("A") + i) for i in range(26)]

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two 3D points"""
    return np.sqrt(np.sum((p1 - p2) ** 2))

def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)

def extract_engineered_features(landmarks):
    """
    Extract hand-size-invariant features from raw landmarks
    MUST match the feature extraction in training script
    """
    points = landmarks.reshape(21, 3)

    wrist = points[0]
    centered = points - wrist

    palm_size = calculate_distance(points[0], points[9]) + 1e-8
    normalized = centered / palm_size
    normalized_coords = normalized.flatten()

    finger_tips = [4, 8, 12, 16, 20]
    finger_bases = [2, 5, 9, 13, 17]

    finger_lengths = []
    for tip, base in zip(finger_tips, finger_bases):
        length = calculate_distance(points[tip], points[base]) / palm_size
        finger_lengths.append(length)

    fingertip_distances = []
    for i in range(len(finger_tips)):
        for j in range(i+1, len(finger_tips)):
            dist = calculate_distance(points[finger_tips[i]], points[finger_tips[j]]) / palm_size
            fingertip_distances.append(dist)

    finger_angles = []
    finger_angles.append(calculate_angle(points[1], points[2], points[3]))
    finger_angles.append(calculate_angle(points[2], points[3], points[4]))
    for base in [5, 9, 13, 17]:
        finger_angles.append(calculate_angle(points[base], points[base+1], points[base+2]))
        finger_angles.append(calculate_angle(points[base+1], points[base+2], points[base+3]))

    palm_angles = []
    for tip in finger_tips:
        angle = calculate_angle(points[9], points[0], points[tip])
        palm_angles.append(angle)

    finger_spread = []
    mcp_points = [1, 5, 9, 13, 17]
    for i in range(len(mcp_points)-1):
        angle = calculate_angle(points[mcp_points[i]], points[0], points[mcp_points[i+1]])
        finger_spread.append(angle)

    all_features = np.concatenate([
        normalized_coords,
        finger_lengths,
        fingertip_distances,
        finger_angles,
        palm_angles,
        finger_spread
    ])

    return all_features

print(f"Loading model from: {MODEL_PATH}")
clf = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_action_time = 0.0
ACTION_COOLDOWN = 2

subprocess.Popen(["notepad.exe"])
time.sleep(1.5)

notepad = None
for w in gw.getAllTitles():
    if "Notepad" in w:
        notepad = gw.getWindowsWithTitle(w)[0]
        break

if notepad is not None:
    notepad.activate()
    time.sleep(0.3)

if not cap.isOpened():
    print("Cannot open camera.")
    exit()

print("\n" + "="*60)
print("Sign Language Recognition Active")
print("="*60)
if USE_IMPROVED_MODEL:
    print("Using: IMPROVED MODEL (person-independent, feature-engineered)")
    print("Note: This model generalizes better to new users!")
else:
    print("Using: ORIGINAL MODEL (may not generalize well)")
print("\nInstructions:")
print("  - Show hand gestures for letters A-Z")
print("  - Recognized letters will be typed into Notepad")
print("  - Press 'q' to quit")
print("  - Press 'c' to show confidence scores")
print("="*60 + "\n")

show_confidence = False

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture_text = "No gesture"
        confidence = 0.0
        top_predictions = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                coords_array = np.array(coords)

                if USE_IMPROVED_MODEL:
                    X = extract_engineered_features(coords_array).reshape(1, -1)
                else:
                    X = coords_array.reshape(1, -1)

                pred_label = int(clf.predict(X)[0])
                gesture_text = letters[pred_label]

                if hasattr(clf, 'predict_proba'):
                    proba = clf.predict_proba(X)[0]
                    confidence = proba[pred_label]

                    top_idx = np.argsort(proba)[-3:][::-1]
                    top_predictions = [(letters[i], proba[i]) for i in top_idx]

                now = time.time()
                if (now - last_action_time) > ACTION_COOLDOWN:
                    char = gesture_text.lower()

                    if notepad is not None:
                        try:
                            notepad.activate()
                            time.sleep(0.1)
                        except:
                            pass

                    pyautogui.write(char)
                    print(f"✓ Typed: '{char}' (confidence: {confidence:.2%})")

                    last_action_time = now


        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if show_confidence and top_predictions:
            y_offset = 70
            cv2.putText(frame, "Top predictions:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for i, (letter, prob) in enumerate(top_predictions):
                y_offset += 25
                color = (0, 255, 0) if i == 0 else (200, 200, 200)
                text = f"{i+1}. {letter}: {prob:.1%}"
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        model_type = "IMPROVED" if USE_IMPROVED_MODEL else "ORIGINAL"
        cv2.putText(frame, f"Model: {model_type}", (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.putText(frame, "Press 'q' to quit, 'c' for confidence", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("ASL Sign Language Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            show_confidence = not show_confidence
            print(f"Confidence display: {'ON' if show_confidence else 'OFF'}")

cap.release()
cv2.destroyAllWindows()

print("\nDemo closed.")