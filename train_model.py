import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Paths
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
csv_file = os.path.join(desktop_path, "combined_gestures.csv")
model_file = os.path.join(desktop_path, "gesture_model.pkl")
confusion_matrix_file = os.path.join(desktop_path, "confusion_matrix.png")

# Read dataset
df = pd.read_csv(csv_file)

print("Dataset shape:", df.shape)
print(df.head())

# Features and target
X = df.drop("command", axis=1)
y = df["command"]

# Convert hand_side text to numbers
X["hand_side"] = X["hand_side"].map({
    "Left": 0,
    "Right": 1,
    "Unknown": 2
})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Classification report in console
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()
plt.title("Confusion Matrix - Gesture Recognition Model")
plt.tight_layout()

# Save confusion matrix image
plt.savefig(confusion_matrix_file, dpi=300)
plt.close()

print("\nConfusion matrix saved to:", confusion_matrix_file)

# Save trained model
joblib.dump(model, model_file)
print("\nModel saved to:", model_file)