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
MODEL_PATH = MODELS_DIR / "sign_language_model.joblib"

letters = [chr(ord("A") + i) for i in range(26)]
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
    print("Kamera se ne može otvoriti.")
    exit()

print("Prepoznavanje slova aktivno. Notepad je otvoren.")
print("Svaka gesta A-Z automatski se pretvara u malo slovo.")

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

        gesture_text = "Nema geste"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                X = np.array(coords).reshape(1, -1)
                pred_label = int(clf.predict(X)[0])
                gesture_text = letters[pred_label]

                now = time.time()

                if (now - last_action_time) > ACTION_COOLDOWN:
                    char = gesture_text.lower()

                    if notepad is not None:
                        notepad.activate()
                        time.sleep(0.1)

                    pyautogui.write(char)

                    print("Upisano slovo:", char)

                    last_action_time = now

                break

        cv2.putText(frame, f"Gesta: {gesture_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("ASL Prepoznavanje Slova", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
