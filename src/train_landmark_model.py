from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "hand_landmarks.csv"
MODEL_PATH = MODELS_DIR / "landmark_model.joblib"

df = pd.read_csv(CSV_PATH)

X = df.drop(columns=["label"]).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)

print("Točnost (Accuracy):", accuracy)
print("Makro preciznost (Precision):", precision)
print("Makro odziv (Recall):", recall)
print("Makro F1-mjera (F1-score):", f1)
print()

print("Izvještaj po klasama:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Konfuzijska matrica:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(clf, MODEL_PATH)
print("Model spremljen u:", MODEL_PATH)
