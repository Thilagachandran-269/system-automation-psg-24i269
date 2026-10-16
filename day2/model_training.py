
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# ==================================================
# 1. FILE PATHS AND SENSOR FEATURES
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_FILE = BASE_DIR / "Sensor_history.csv"
MODEL_FILE = BASE_DIR / "extra_trees_sensor_model.pkl"

FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "current",
    "rpm"
]

TARGET = "state"


# ==================================================
# 2. LOAD DATASET
# ==================================================

print("=" * 55)
print("       MACHINE FAULT DETECTION - EXTRA TREES")
print("=" * 55)

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {CSV_FILE}"
    )

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip().str.lower()

required_columns = FEATURES + [TARGET]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns in CSV: {missing}"
    )

df = df[required_columns].copy()


# ==================================================
# 3. CLEAN SENSOR DATA
# ==================================================

for col in FEATURES:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

df[TARGET] = (
    df[TARGET]
    .astype("string")
    .str.strip()
    .str.upper()
)

# Correct a known truncated state label
df[TARGET] = df[TARGET].replace({
    "MOTOR_OVERLO": "MOTOR_OVERLOAD"
})

# Remove incomplete records
df = df.dropna(subset=required_columns)
df = df[df[TARGET] != ""]

if len(df) < 10:
    raise ValueError(
        "Not enough valid sensor records."
    )

print("\nDataset loaded successfully")
print("Valid records:", len(df))

print("\nMachine state distribution:")
print(df[TARGET].value_counts())


# ==================================================
# 4. PREPARE FEATURES AND TARGET
# ==================================================

X = df[FEATURES]
y = df[TARGET]

if y.nunique() < 2:
    raise ValueError(
        "Dataset must contain at least two states."
    )

if y.value_counts().min() < 2:
    raise ValueError(
        "Each state needs at least two records "
        "for a stratified train/test split."
    )


# ==================================================
# 5. SPLIT DATA
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==================================================
# 6. CREATE EXTRA TREES MODEL
# ==================================================

model = ExtraTreesClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ==================================================
# 7. TRAIN MODEL
# ==================================================

print("\nTraining Extra Trees model...")

model.fit(X_train, y_train)

print("Training completed successfully!")


# ==================================================
# 8. EVALUATE MODEL
# ==================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 55)
print("                 MODEL RESULTS")
print("=" * 55)

print(f"Test accuracy : {accuracy * 100:.2f}%")
print(f"Macro F1-score: {macro_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=model.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print("Class order:", list(model.classes_))

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )
)


# ==================================================
# 9. SENSOR FEATURE IMPORTANCE
# ==================================================

print("\n" + "=" * 55)
print("             SENSOR FEATURE IMPORTANCE")
print("=" * 55)

importance_df = pd.DataFrame({
    "Sensor": FEATURES,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.to_string(index=False))


# ==================================================
# 10. SAVE TRAINED MODEL
# ==================================================

joblib.dump(model, MODEL_FILE)

print("\nTrained model saved successfully!")
print("File:", MODEL_FILE)


# ==================================================
# 11. PREDICT NEW MACHINE SENSOR READINGS
# ==================================================

print("\n" + "=" * 55)
print("           NEW MACHINE STATE PREDICTION")
print("=" * 55)

try:
    temperature = float(input("Temperature: "))
    pressure = float(input("Pressure: "))
    vibration = float(input("Vibration: "))
    current = float(input("Current: "))
    rpm = float(input("RPM: "))

    new_data = pd.DataFrame(
        [[temperature, pressure, vibration, current, rpm]],
        columns=FEATURES
    )

    prediction = model.predict(new_data)[0]
    probabilities = model.predict_proba(new_data)[0]

    print("\nPredicted Machine State:", prediction)

    print("\nPredicted class probabilities:")

    for state, probability in zip(
        model.classes_, probabilities
    ):
        print(f"{state}: {probability * 100:.2f}%")

    confidence = max(probabilities) * 100

    print(f"\nHighest class probability: {confidence:.2f}%")

    if prediction == "NORMAL":
        print("Status: No fault indicated by the model.")
    else:
        print("Status: Potential machine fault detected!")
        print("Investigate the sensor readings.")

except ValueError:
    print(
        "\nInvalid input. Please enter numeric "
        "values for all five sensors."
    )

print("\nProgram finished.")
