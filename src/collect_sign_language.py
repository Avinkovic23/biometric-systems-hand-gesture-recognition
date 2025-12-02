from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

USER_ID = input("Unesi ID korisnika (npr. ime ili nadimak): ").strip() or "user"
OUTPUT_CSV = DATA_DIR / f"hand_landmarks_{USER_ID}.csv"

CAPTURE_INTERVAL = 0.4

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ne mogu otvoriti kameru")
    exit()

all_rows = []

auto_capture = False
current_letter = None
current_label_idx = None
last_capture_time = 0.0

sample_counts = {chr(ord("A") + i): 0 for i in range(26)}

print("Kontrole:")
print("a-z = odaberi slovo (gestu)")
print("Space = pokreni/pauziraj automatsko snimanje")
print("ESC = izlaz")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    last_landmarks = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        last_landmarks = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.append(lm.x)
                    coords.append(lm.y)
                    coords.append(lm.z)
                last_landmarks = coords
                break

        status_text = f"Auto: {'ON' if auto_capture else 'OFF'}"
        gesture_text = f"Gesta: {current_letter}" if current_letter is not None else "Gesta: -"
        info_text = "a-z: odaberi slovo  |  Space: start/pauza  |  ESC: izlaz"

        cv2.putText(
            frame,
            status_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            gesture_text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            info_text,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        if current_letter is not None:
            count = sample_counts[current_letter]
            cv2.putText(
                frame,
                f"Broj uzoraka za {current_letter}: {count}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow("Prikupljanje gesti A-Z", frame)

        now = time.time()
        if auto_capture and current_label_idx is not None and last_landmarks is not None:
            if now - last_capture_time >= CAPTURE_INTERVAL:
                row = last_landmarks + [current_label_idx, current_letter, USER_ID]
                all_rows.append(row)
                sample_counts[current_letter] += 1
                last_capture_time = now
                print(
                    f"Skeniran uzorak za slovo {current_letter} broj "
                    f"{sample_counts[current_letter]} (user: {USER_ID})"
                )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if ord("a") <= key <= ord("z"):
            letter = chr(key).upper()
            current_letter = letter
            current_label_idx = ord(letter) - ord("A")
            print(f"Aktivna gesta: {current_letter} (label {current_label_idx})")

        if key == 32:
            auto_capture = not auto_capture
            state = "POKRENUTO" if auto_capture else "PAUZIRANO"
            print(f"Automatsko skeniranje: {state}")

cap.release()
cv2.destroyAllWindows()

if all_rows:
    num_coords = len(all_rows[0]) - 3
    num_points = num_coords // 3

    columns = []
    for i in range(num_points):
        columns.append(f"x{i}")
        columns.append(f"y{i}")
        columns.append(f"z{i}")
    columns.append("label")
    columns.append("letter")
    columns.append("user_id")

    df_new = pd.DataFrame(all_rows, columns=columns)

    if OUTPUT_CSV.exists():
        df_old = pd.read_csv(OUTPUT_CSV)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"Spremljeno ukupno {len(df_all)} uzoraka u {OUTPUT_CSV}")
    print("Broj uzoraka po slovu u ovoj sesiji:")
    for letter, count in sample_counts.items():
        if count > 0:
            print(f"{letter}: {count}")
else:
    print("Nema prikupljenih uzoraka.")
