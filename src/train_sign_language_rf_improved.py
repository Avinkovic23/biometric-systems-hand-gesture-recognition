from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "hand_landmarks_merged.csv"
MODEL_PATH = MODELS_DIR / "sign_language_model_improved.joblib"


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

    Landmarks indexing (MediaPipe):
    0: WRIST
    1-4: THUMB (CMC, MCP, IP, TIP)
    5-8: INDEX (MCP, PIP, DIP, TIP)
    9-12: MIDDLE (MCP, PIP, DIP, TIP)
    13-16: RING (MCP, PIP, DIP, TIP)
    17-20: PINKY (MCP, PIP, DIP, TIP)
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

print("Loading data...")
df = pd.read_csv(CSV_PATH)

feature_cols = [c for c in df.columns if c.startswith("x") or c.startswith("y") or c.startswith("z")]
X_raw = df[feature_cols].values
y = df["label"].values
users = df["user_id"].values

print(f"Total samples: {len(df)}")
print(f"Users: {np.unique(users)}")
print(f"Classes: {np.unique(y)} (Letters: {len(np.unique(y))})")

print("\nExtracting engineered features...")
X_engineered = np.array([extract_engineered_features(sample) for sample in X_raw])
print(f"Feature shape: {X_engineered.shape} ({X_engineered.shape[1]} features per sample)")

print("\n" + "="*60)
print("PERSON-INDEPENDENT CROSS-VALIDATION (Leave-One-User-Out)")
print("="*60)

logo = LeaveOneGroupOut()
fold_scores = []
fold_results = []

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_engineered, y, groups=users)):
    X_train, X_test = X_engineered[train_idx], X_engineered[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    test_user = users[test_idx][0]

    print(f"\nFold {fold_idx + 1}: Testing on user '{test_user}'")
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    fold_scores.append({
        'user': test_user,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

    fold_results.append({
        'user': test_user,
        'y_test': y_test,
        'y_pred': y_pred
    })

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")

print("\n" + "="*60)
print("AGGREGATE CROSS-VALIDATION RESULTS")
print("="*60)

scores_df = pd.DataFrame(fold_scores)
print(scores_df.to_string(index=False))

print(f"\nMean ± Std:")
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    mean = scores_df[metric].mean()
    std = scores_df[metric].std()
    print(f"  {metric.capitalize()}: {mean:.4f} ± {std:.4f}")

print("\n" + "="*60)
print("HYPERPARAMETER TUNING (GridSearchCV)")
print("="*60)

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

base_rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

from sklearn.model_selection import GroupKFold
group_kfold = GroupKFold(n_splits=2)

grid_search = GridSearchCV(
    base_rf,
    param_grid,
    cv=group_kfold,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

print("Starting grid search (this may take a while)...")
grid_search.fit(X_engineered, y, groups=users)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

print("\n" + "="*60)
print("TRAINING FINAL MODEL ON ALL DATA")
print("="*60)

best_clf = grid_search.best_estimator_
best_clf.fit(X_engineered, y)

feature_importances = best_clf.feature_importances_
top_features_idx = np.argsort(feature_importances)[-20:][::-1]

print("\nTop 20 Most Important Features:")
feature_names = (
    [f"norm_{col}" for col in feature_cols] +
    [f"finger_len_{i}" for i in range(5)] +
    [f"fingertip_dist_{i}" for i in range(10)] +
    [f"finger_angle_{i}" for i in range(10)] +
    [f"palm_angle_{i}" for i in range(5)] +
    [f"finger_spread_{i}" for i in range(4)]
)

for idx in top_features_idx:
    print(f"  {feature_names[idx]}: {feature_importances[idx]:.4f}")

joblib.dump(best_clf, MODEL_PATH)
print(f"\nImproved model saved to: {MODEL_PATH}")

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT (Aggregated Test Sets)")
print("="*60)

all_y_test = np.concatenate([fr['y_test'] for fr in fold_results])
all_y_pred = np.concatenate([fr['y_pred'] for fr in fold_results])

letters = [chr(ord("A") + i) for i in range(26)]
print(classification_report(all_y_test, all_y_pred, target_names=letters, zero_division=0))

print("\nGenerating confusion matrix visualization...")
cm = confusion_matrix(all_y_test, all_y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=letters, yticklabels=letters,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Person-Independent Cross-Validation', fontsize=16)
plt.xlabel('Predicted Letter', fontsize=12)
plt.ylabel('True Letter', fontsize=12)
plt.tight_layout()

confusion_matrix_path = RESULTS_DIR / "confusion_matrix_improved.png"
plt.savefig(confusion_matrix_path, dpi=300)
print(f"Confusion matrix saved to: {confusion_matrix_path}")

summary_path = RESULTS_DIR / "training_summary_improved.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("IMPROVED SIGN LANGUAGE MODEL - TRAINING SUMMARY\n")
    f.write("="*60 + "\n\n")

    f.write("Configuration:\n")
    f.write(f"  - Person-independent evaluation: Leave-One-User-Out CV\n")
    f.write(f"  - Feature engineering: Normalized coords + angles + distances\n")
    f.write(f"  - Total features: {X_engineered.shape[1]}\n")
    f.write(f"  - Total samples: {len(df)}\n")
    f.write(f"  - Number of users: {len(np.unique(users))}\n")
    f.write(f"  - Number of classes: {len(np.unique(y))}\n\n")

    f.write("Cross-Validation Results:\n")
    f.write(scores_df.to_string(index=False) + "\n\n")

    f.write("Aggregate Metrics (Mean ± Std):\n")
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        mean = scores_df[metric].mean()
        std = scores_df[metric].std()
        f.write(f"  {metric.capitalize()}: {mean:.4f} ± {std:.4f}\n")

    f.write(f"\nBest Hyperparameters:\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"  {param}: {value}\n")

    f.write(f"\nBest CV Score: {grid_search.best_score_:.4f}\n")

print(f"\nTraining summary saved to: {summary_path}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nModel saved: {MODEL_PATH}")
print(f"Results saved: {RESULTS_DIR}")
print("\nNOTE: The person-independent accuracy is the TRUE measure of")
print("how well your model will perform on new users in the live demo.")
