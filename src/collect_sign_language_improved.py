from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from collections import deque

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

USER_ID = input("Unesi ID korisnika: ").strip() or "user"
OUTPUT_CSV = DATA_DIR / f"hand_landmarks_{USER_ID}.csv"

TARGET_SAMPLES_PER_LETTER = int(input("Broj uzoraka po slovu (default 500): ").strip() or "500")

CAPTURE_INTERVAL = 0.5
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
STABILITY_THRESHOLD = 0.02
STABILITY_FRAMES = 5

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ne mogu otvoriti kameru")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

all_rows = []
auto_capture = False
current_letter = None
current_label_idx = None
last_capture_time = 0.0

letters = [chr(ord("A") + i) for i in range(26)]
current_letter_index = 0
sample_counts = {letter: 0 for letter in letters}

landmark_history = deque(maxlen=STABILITY_FRAMES)

def calculate_hand_stability(landmarks_history):
    if len(landmarks_history) < STABILITY_FRAMES:
        return False, 0.0

    positions = np.array(landmarks_history)
    variance = np.var(positions, axis=0)
    mean_variance = np.mean(variance)

    is_stable = mean_variance < STABILITY_THRESHOLD
    return is_stable, mean_variance

def calculate_hand_quality(hand_landmarks, frame_shape, quality_threshold=0.6):
    points = []
    for lm in hand_landmarks.landmark:
        x = int(lm.x * frame_shape[1])
        y = int(lm.y * frame_shape[0])
        points.append((x, y, lm.z))

    all_x = [p[0] for p in points]
    all_y = [p[1] for p in points]

    hand_width = max(all_x) - min(all_x)
    hand_height = max(all_y) - min(all_y)

    min_size = min(frame_shape[0], frame_shape[1]) * 0.3
    max_size = min(frame_shape[0], frame_shape[1]) * 0.9

    size_ok = min_size < hand_width < max_size and min_size < hand_height < max_size

    center_x = np.mean(all_x)
    center_y = np.mean(all_y)

    margin_x = frame_shape[1] * 0.15
    margin_y = frame_shape[0] * 0.15

    position_ok = (margin_x < center_x < frame_shape[1] - margin_x and
                   margin_y < center_y < frame_shape[0] - margin_y)

    visibility_ok = all(lm.visibility > 0.5 for lm in hand_landmarks.landmark if hasattr(lm, 'visibility'))

    quality_score = sum([size_ok, position_ok, visibility_ok]) / 3.0

    return quality_score > quality_threshold, quality_score

def draw_progress_bar(frame, x, y, width, height, progress, color):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    filled_width = int(width * progress)
    if filled_width > 0:
        cv2.rectangle(frame, (x, y), (x + filled_width, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)

print("\n" + "="*70)
print("KONTROLE:")
print("="*70)
print("Space = Start/Stop automatsko snimanje")
print("n     = Sljedece slovo")
print("p     = Prethodno slovo")
print("r     = Reset brojaca za trenutno slovo")
print("t     = Toggle kvaliteta (0.6 <-> 0.3)")
print("ESC   = Izlaz i spremi")
print("="*70 + "\n")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
) as hands:

    current_letter = letters[current_letter_index]
    current_label_idx = current_letter_index

    last_frame_time = time.time()
    fps = 0

    countdown_start = None
    countdown_duration = 0

    low_quality_mode = False
    quality_threshold = 0.6

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        current_time = time.time()
        fps = 1.0 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
        last_frame_time = current_time

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hand_detected = False
        hand_stable = False
        hand_quality_ok = False
        quality_score = 0.0
        stability_score = 0.0
        current_landmarks = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                current_landmarks = coords
                hand_detected = True

                hand_quality_ok, quality_score = calculate_hand_quality(hand_landmarks, frame.shape, quality_threshold)

                landmark_history.append(coords[:3])
                hand_stable, stability_score = calculate_hand_stability(landmark_history)

                break
        else:
            landmark_history.clear()

        overlay = frame.copy()

        panel_height = 180
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y_offset = 30

        cv2.putText(frame, f"Slovo: {current_letter} ({current_letter_index + 1}/26)",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        y_offset += 35
        progress = sample_counts[current_letter] / TARGET_SAMPLES_PER_LETTER
        draw_progress_bar(frame, 20, y_offset - 20, 400, 25, min(progress, 1.0), (0, 255, 0))
        cv2.putText(frame, f"{sample_counts[current_letter]}/{TARGET_SAMPLES_PER_LETTER}",
                    (430, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_offset += 40
        status_color = (0, 255, 0) if auto_capture else (100, 100, 100)
        status_text = "SNIMANJE" if auto_capture else "PAUZIRANO"
        cv2.putText(frame, f"Status: {status_text}",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        y_offset += 30
        hand_text = "DETEKTIRANA" if hand_detected else "NIJE DETEKTIRANA"
        hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, f"Ruka: {hand_text}",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)

        quality_mode_text = "LOW" if low_quality_mode else "HIGH"
        quality_text = f"Kvaliteta: {quality_score:.0%} ({quality_mode_text})"
        quality_color = (0, 255, 0) if hand_quality_ok else (0, 165, 255)
        cv2.putText(frame, quality_text,
                    (20, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)

        stability_text = f"Stabilnost: {'OK' if hand_stable else 'DRZI'}"
        stability_color = (0, 255, 0) if hand_stable else (0, 165, 255)
        cv2.putText(frame, stability_text,
                    (250, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 2)

        cv2.rectangle(frame, (0, h - 80), (w, h), (40, 40, 40), -1)
        cv2.putText(frame, "Space: Start/Stop | N: Next | P: Prev | R: Reset | T: Quality | ESC: Exit",
                    (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        total_samples = sum(sample_counts.values())
        total_progress = total_samples / (26 * TARGET_SAMPLES_PER_LETTER)
        cv2.putText(frame, f"Ukupno: {total_samples}/{26 * TARGET_SAMPLES_PER_LETTER} ({total_progress:.0%})",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if auto_capture and countdown_start is None:
            countdown_start = time.time()

        if countdown_start is not None:
            elapsed = time.time() - countdown_start
            remaining = max(0, countdown_duration - elapsed)

            if remaining > 0:
                countdown_text = f"{int(remaining) + 1}"
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
                text_x = (w - text_size[0]) // 2
                text_y = (h + text_size[1]) // 2

                cv2.putText(frame, countdown_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)
            else:
                countdown_start = None

        if (auto_capture and
            countdown_start is None and
            current_landmarks is not None and
            hand_detected and
            hand_quality_ok and
            hand_stable and
            sample_counts[current_letter] < TARGET_SAMPLES_PER_LETTER):

            if time.time() - last_capture_time >= CAPTURE_INTERVAL:
                row = current_landmarks + [current_label_idx, current_letter, USER_ID]
                all_rows.append(row)
                sample_counts[current_letter] += 1
                last_capture_time = time.time()

                print(f"[{current_letter}] Uzorak {sample_counts[current_letter]}/{TARGET_SAMPLES_PER_LETTER}")

                if sample_counts[current_letter] >= TARGET_SAMPLES_PER_LETTER:
                    print(f"\n✓ Slovo {current_letter} zavrseno! ({TARGET_SAMPLES_PER_LETTER} uzoraka)")

                    if current_letter_index < 25:
                        current_letter_index += 1
                        current_letter = letters[current_letter_index]
                        current_label_idx = current_letter_index
                        print(f"→ Prelazim na slovo: {current_letter}\n")
                        countdown_start = time.time()
                    else:
                        print("\n✓✓✓ SVI UZORCI PRIKUPLJENI! ✓✓✓")
                        auto_capture = False

        cv2.imshow("Prikupljanje Gesti - Poboljsano", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key == 32:
            auto_capture = not auto_capture
            countdown_start = time.time() if auto_capture else None
            state = "POKRENUTO" if auto_capture else "PAUZIRANO"
            print(f"\nAutomatsko snimanje: {state}")

        if key == ord('n'):
            if current_letter_index < 25:
                current_letter_index += 1
                current_letter = letters[current_letter_index]
                current_label_idx = current_letter_index
                print(f"\n→ Slovo: {current_letter}")

        if key == ord('p'):
            if current_letter_index > 0:
                current_letter_index -= 1
                current_letter = letters[current_letter_index]
                current_label_idx = current_letter_index
                print(f"\n← Slovo: {current_letter}")

        if key == ord('r'):
            sample_counts[current_letter] = 0
            print(f"\nReset brojaca za slovo {current_letter}")

        if key == ord('t'):
            low_quality_mode = not low_quality_mode
            quality_threshold = 0.3 if low_quality_mode else 0.6
            mode_text = "LOW (0.3)" if low_quality_mode else "HIGH (0.6)"
            print(f"\nKvaliteta mod: {mode_text}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("SPREMANJE PODATAKA...")
print("="*70)

if all_rows:
    num_coords = len(all_rows[0]) - 3
    num_points = num_coords // 3

    columns = []
    for i in range(num_points):
        columns.extend([f"x{i}", f"y{i}", f"z{i}"])
    columns.extend(["label", "letter", "user_id"])

    df_new = pd.DataFrame(all_rows, columns=columns)

    if OUTPUT_CSV.exists():
        df_old = pd.read_csv(OUTPUT_CSV)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✓ Spremljeno: {OUTPUT_CSV}")
    print(f"✓ Ukupno uzoraka: {len(df_all)}")
    print(f"✓ Novih uzoraka: {len(df_new)}")

    print("\nUzorci po slovu u ovoj sesiji:")
    for letter in letters:
        count = sample_counts[letter]
        if count > 0:
            status = "✓" if count >= TARGET_SAMPLES_PER_LETTER else f"{count}/{TARGET_SAMPLES_PER_LETTER}"
            print(f"  {letter}: {status}")

    print("\n" + "="*70)
    print("ZAVRSENO!")
    print("="*70)
else:
    print("\nNema prikupljenih uzoraka.")

