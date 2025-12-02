from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "data" / "hand_gestures" / "test" / "test"
MODELS_DIR = BASE_DIR / "models"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

model_path = MODELS_DIR / "gesture_cnn.h5"
model = tf.keras.models.load_model(model_path)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

y_true = test_generator.classes

pred_probs = model.predict(test_generator)
y_pred = np.argmax(pred_probs, axis=1)

class_indices = test_generator.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

cm = confusion_matrix(y_true, y_pred)

print("Točnost (Accuracy):", accuracy)
print("Makro preciznost (Precision):", precision)
print("Makro odziv (Recall):", recall)
print("Makro F1-mjera (F1-score):", f1)
print("\nIzvještaj po klasama:")
print(classification_report(y_true, y_pred, target_names=target_names))
print("Konfuzijska matrica (Confusion matrix):")
print(cm)
