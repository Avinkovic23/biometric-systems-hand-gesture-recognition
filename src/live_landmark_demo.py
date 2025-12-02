from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import subprocess
import time

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "landmark_model.joblib"

gestures = {
    0: "Palac gore",
    1: "V znak",
    2: "Palac dolje",
    3: "Tri prsta",
    4: "Telefon znak"
}

calc_keys = {
    0: "add",
    1: "2",
    2: "subtract",
    3: "3",
    4: "enter"
}

text_keys = {
    0: "a",
    1: "b",
    2: "c",
    3: "d",
    4: "backspace"
}

clf = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

current_mode = None
last_mode_switch_time = 0.0
last_action_time = 0.0
MODE_SWITCH_COOLDOWN = 3.0
APP_START_DELAY = 2.0
ACTION_COOLDOWN = 1.5

if not cap.isOpened():
    print("Ne mogu otvoriti kameru")
    exit()

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

        gesture_text = "Nema detektirane geste"
        mode_text = "Mode: nijedan"

        if current_mode == "calc":
            mode_text = "Mode: kalkulator"
        elif current_mode == "text":
            mode_text = "Mode: tekst"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.append(lm.x)
                    coords.append(lm.y)
                    coords.append(lm.z)

                X = np.array(coords, dtype=np.float32).reshape(1, -1)
                pred_label = int(clf.predict(X)[0])
                gesture_text = gestures.get(pred_label, f"Gesta {pred_label}")

                now = time.time()

                if current_mode is None:
                    if pred_label == 0 and (now - last_mode_switch_time) > MODE_SWITCH_COOLDOWN:
                        subprocess.Popen(["calc.exe"])
                        current_mode = "calc"
                        last_mode_switch_time = now
                    elif pred_label == 2 and (now - last_mode_switch_time) > MODE_SWITCH_COOLDOWN:
                        subprocess.Popen(["notepad.exe"])
                        current_mode = "text"
                        last_mode_switch_time = now
                else:
                    if (now - last_action_time) > ACTION_COOLDOWN and (now - last_mode_switch_time) > APP_START_DELAY:
                        if current_mode == "calc" and pred_label in calc_keys:
                            key = calc_keys[pred_label]
                            pyautogui.press(key)
                            last_action_time = now
                        elif current_mode == "text" and pred_label in text_keys:
                            key = text_keys[pred_label]
                            if key == "backspace":
                                pyautogui.press("backspace")
                            else:
                                pyautogui.press(key)
                            last_action_time = now

                break

        cv2.putText(
            frame,
            mode_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            gesture_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Prepoznavanje gesti (landmarks)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
