import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb

from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# ================= SETTINGS =================

DATA_FOLDER = os.getenv("DATA_FOLDER", "data/raw")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "trained_models")
LABEL_COLUMN = "Label"
RANDOM_STATE = 42

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================= LOAD DATA =================

csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
df_list = []

print(f"[+] Found {len(csv_files)} CSV files")

for file in tqdm(csv_files, desc="Loading CSV"):
    try:
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip()

        if LABEL_COLUMN not in df.columns:
            print(f"[!] Skipping (no label): {file}")
            continue

        df_list.append(df)

    except Exception as e:
        print(f"[!] Error reading {file}: {e}")

if not df_list:
    raise ValueError("❌ No valid CSV files loaded. Check DATA_FOLDER.")

data = pd.concat(df_list, ignore_index=True)
print(f"[+] Loaded {len(data)} rows")

# ================= PREPROCESS =================

data[LABEL_COLUMN] = data[LABEL_COLUMN].apply(
    lambda x: 0 if str(x).upper() == "BENIGN" else 1
)

non_numeric_cols = data.select_dtypes(include=["object"]).columns.tolist()
non_numeric_cols = [c for c in non_numeric_cols if c != LABEL_COLUMN]
data.drop(columns=non_numeric_cols, inplace=True)

data.replace([np.inf, -np.inf], 0, inplace=True)
data.fillna(0, inplace=True)

X = data.drop(columns=[LABEL_COLUMN])
y = data[LABEL_COLUMN]

feature_columns = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("[+] Applying SMOTE...")
smote = SMOTE(random_state=RANDOM_STATE)
X_res, y_res = smote.fit_resample(X_scaled, y)

print(f"[+] Balanced counts: {np.bincount(y_res)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, stratify=y_res, random_state=RANDOM_STATE
)

# ================= MODEL DEFINITIONS =================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
)

lgbm_model = lgb.LGBMClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
)

stack_model = StackingClassifier(
    estimators=[
        ("rf", rf_model),
        ("xgb", xgb_model),
        ("lgbm", lgbm_model)
    ],
    final_estimator=RandomForestClassifier(
        n_estimators=120,
        random_state=RANDOM_STATE
    ),
    cv=5
)

models = {
    "RandomForest": rf_model,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model,
    "Stacking": stack_model
}

results = []

plt.figure(figsize=(8, 6))

# ================= TRAIN =================

for name, model in models.items():
    print(f"\n[+] Training {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1       : {f1:.4f}")
    print(f"   ROC-AUC  : {roc_auc:.4f}")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": roc_auc
    })

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{name}_confusion_matrix.png"))
    plt.clf()

    # Feature importance (only if supported)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:20]

        plt.bar(range(len(idx)), importances[idx])
        plt.xticks(range(len(idx)), [feature_columns[i] for i in idx], rotation=90)
        plt.title(f"{name} - Top Features")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"{name}_features.png"))
        plt.clf()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} ({roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "ROC.png"))
plt.clf()

# ================= SAVE ARTIFACTS =================

for name, model in models.items():
    joblib.dump(model, os.path.join(OUTPUT_FOLDER, f"{name}.pkl"))

joblib.dump(scaler, os.path.join(OUTPUT_FOLDER, "scaler.pkl"))
joblib.dump(feature_columns, os.path.join(OUTPUT_FOLDER, "feature_columns.pkl"))

pd.DataFrame(results).to_csv(
    os.path.join(OUTPUT_FOLDER, "model_metrics.csv"),
    index=False
)

print("\n[+] Training complete")
print(f"[+] Artifacts saved to: {OUTPUT_FOLDER}")
