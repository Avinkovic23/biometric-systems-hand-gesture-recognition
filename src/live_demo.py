from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
TRAIN_DIR = BASE_DIR / "data" / "hand_gestures" / "train" / "train"

IMAGE_SIZE = (128, 128)

model_path = MODELS_DIR / "gesture_cnn.h5"
model = tf.keras.models.load_model(model_path)

temp_datagen = ImageDataGenerator(rescale=1.0 / 255)
temp_generator = temp_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)

class_indices = temp_generator.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

akcije = {
    "0": "Znak OK",
    "1": "Broj 1 (kažiprst)",
    "2": "Broj 2 (V znak)",
    "3": "Broj 3 (tri prsta)",
    "4": "Broj 4 (četiri prsta)",
    "5": "Otvoreni dlan (pet prstiju)",
    "6": "Tri srednja prsta",
    "7": "Rock znak",
    "8": "I love you znak",
    "9": "Tri prsta - druga varijanta",
    "10": "L znak rukom",
    "11": "Ruka postrani (rub dlana)",
    "12": "Položaj hvata (štipanje)",
    "13": "Uspravna ruka (pokazivanje gore)",
    "14": "Stisnuta šaka",
    "15": "Šaka prema naprijed",
    "16": "Šaka povučena unatrag",
    "17": "Telefon znak",
    "18": "Palac gore",
    "19": "Palac gore postrani"
}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ne mogu otvoriti kameru")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_flipped = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)

    mask_resized = cv2.resize(mask, IMAGE_SIZE)
    mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2RGB)
    mask_norm = mask_rgb.astype("float32") / 255.0
    input_tensor = np.expand_dims(mask_norm, axis=0)

    preds = model.predict(input_tensor, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    class_name = idx_to_class.get(pred_idx, "?")
    opis_geste = akcije.get(class_name, f"Gesta {class_name}")

    text = f"Gesta: {opis_geste}"
    cv2.putText(frame_flipped, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Prepoznavanje gesti", frame_flipped)
    cv2.imshow("Maska za model", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
