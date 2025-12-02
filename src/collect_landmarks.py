from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = DATA_DIR / "hand_landmarks.csv"

gestures = {
    0: "palac_gore",
    1: "v_znak",
    2: "palac_dolje",
    3: "tri_prsta",
    4: "telefon_znak"
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ne mogu otvoriti kameru")
    exit()

all_rows = []

print("Pritisni tipke 0–4 za spremanje uzorka trenutne geste.")
print("0 = palac gore, 1 = V znak, 2 = palac dolje, 3 = 3 prsta, 4 = telefon znak")
print("Pritisni q za izlaz.")

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

        cv2.putText(
            frame,
            "0: palac gore  1: V  2: palac dolje  3: tri prsta  4: telefon  |  q: izlaz",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        cv2.imshow("Prikupljanje gesti (landmarks)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in [ord("0"), ord("1"), ord("2"), ord("3"), ord("4")]:
            if last_landmarks is not None:
                label = int(chr(key))
                row = last_landmarks + [label]
                all_rows.append(row)
                print(f"Spremljen uzorak za gestu {label} ({gestures[label]})")

cap.release()
cv2.destroyAllWindows()

if all_rows:
    num_coords = len(all_rows[0]) - 1
    num_points = num_coords // 3

    columns = []
    for i in range(num_points):
        columns.append(f"x{i}")
        columns.append(f"y{i}")
        columns.append(f"z{i}")
    columns.append("label")

    df_new = pd.DataFrame(all_rows, columns=columns)

    if OUTPUT_CSV.exists():
        df_old = pd.read_csv(OUTPUT_CSV)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"Spremljeno ukupno {len(df_all)} uzoraka u {OUTPUT_CSV}")
else:
    print("Nema prikupljenih uzoraka.")
