import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# from ml.data_loader import load_dataset, compute_feature_stats, FEATURE_NAMES
from rainfall_prediction.ml.data_loader import load_dataset, compute_feature_stats, FEATURE_NAMES
# from ml.svm_model import train_svm
# from ml.decision_tree_model import train_decision_tree
# from ml.random_forest_model import train_random_forest
# from ml.metrics import compute_all_metrics
from rainfall_prediction.ml.svm_model import train_svm
from rainfall_prediction.ml.decision_tree_model import train_decision_tree
from rainfall_prediction.ml.random_forest_model import train_random_forest
from rainfall_prediction.ml.metrics import compute_all_metrics

MODELS_SAVE_PATH = os.path.join("models", "trained_models.pkl")


def run_training(csv_path="rainfall_prediction/Rainfall.csv"):
    print("Loading dataset from", csv_path, "...")
    X, y, df = load_dataset(csv_path)
    print(f"  {len(X)} samples loaded | {FEATURE_NAMES.__len__()} features | "
          f"{y.sum()} rain days ({y.mean()*100:.1f}%)")

    # stratify=y ensures both train and test have similar rain/no-rain ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SVM requires scaled features (otherwise large-valued features dominate)
    # DT and RF don't need this — they only use feature orderings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    feature_stats = compute_feature_stats(X, y)

    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    baseline_acc = round(accuracy_score(y_test, dummy.predict(X_test)) * 100, 2)

    rain_pct_train = round(float(y_train.mean()) * 100, 1)
    class_balance = {
        "rain_pct":      rain_pct_train,
        "no_rain_pct":   round(100 - rain_pct_train, 1),
        "total_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "rain_count":    int(y.sum()),
        "no_rain_count": int((y == 0).sum()),
    }

    # ── Train each model ─────────────────────────────────────────
    # SVM uses scaled input; DT and RF use raw input
    print("\nTraining models...")

    print("  [1/3] SVM (RBF kernel)...")
    svm = train_svm(X_train_scaled, y_train)
    svm_pred  = svm.predict(X_test_scaled)
    svm_proba = svm.predict_proba(X_test_scaled)[:, 1]

    print("  [2/3] Decision Tree (calibrated)...")
    dt = train_decision_tree(X_train, y_train)
    dt_pred  = dt.predict(X_test)
    dt_proba = dt.predict_proba(X_test)[:, 1]

    print("  [3/3] Random Forest (200 trees)...")
    rf = train_random_forest(X_train, y_train)
    rf_pred  = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # ── Compute metrics for each ─────────────────────────────────
    print("\nComputing metrics...")

    svm_metrics = compute_all_metrics(y_test, svm_pred, svm_proba, svm, X_train_scaled, y_train)
    dt_metrics  = compute_all_metrics(y_test, dt_pred,  dt_proba,  dt,  X_train,        y_train)
    rf_metrics  = compute_all_metrics(y_test, rf_pred,  rf_proba,  rf,  X_train,        y_train)

    def _log(name, m):
        print(f"  {name}: Acc={m['accuracy']}%  F1={m['f1_score']}%  "
              f"AUC={m['auc_roc']}  MCC={m['mcc']}")

    _log("SVM",           svm_metrics)
    _log("Decision Tree", dt_metrics)
    _log("Random Forest", rf_metrics)

    trained_models = {
        "SVM":           svm,
        "Decision Tree": dt,
        "Random Forest": rf,
    }

    metrics_summary = {
        "SVM":           {"accuracy": svm_metrics["accuracy"], "precision": svm_metrics["precision"],
                          "recall":   svm_metrics["recall"],   "f1_score":  svm_metrics["f1_score"]},
        "Decision Tree": {"accuracy": dt_metrics["accuracy"],  "precision": dt_metrics["precision"],
                          "recall":   dt_metrics["recall"],    "f1_score":  dt_metrics["f1_score"]},
        "Random Forest": {"accuracy": rf_metrics["accuracy"],  "precision": rf_metrics["precision"],
                          "recall":   rf_metrics["recall"],    "f1_score":  rf_metrics["f1_score"]},
    }

    diagnostics = {
        "SVM":           svm_metrics,
        "Decision Tree": dt_metrics,
        "Random Forest": rf_metrics,
        "_meta": {
            "baseline_accuracy": baseline_acc,
            "class_balance":     class_balance,
            "feature_stats":     feature_stats,
            "dataset_source":    "Rainfall.csv (real dataset)",
            "dataset_rows":      int(len(X)),
        },
    }

    # ── Save everything ──────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    payload = {
        "models":      trained_models,
        "scaler":      scaler,
        "metrics":     metrics_summary,
        "diagnostics": diagnostics,
    }
    with open(MODELS_SAVE_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nAll models saved to {MODELS_SAVE_PATH}")
    return trained_models, scaler, metrics_summary, diagnostics


def load_trained_models():
    if not os.path.exists(MODELS_SAVE_PATH):
        return None

    with open(MODELS_SAVE_PATH, "rb") as f:
        data = pickle.load(f)

    model_names = list(data.get("models", {}).keys())
    if "SVM" not in model_names:
        print("Old model file found (no SVM). Will retrain...")
        return None

    diag = data.get("diagnostics", {})
    has_full = diag and "mcc" in diag.get("SVM", {})
    if not has_full:
        print("Incomplete diagnostics found. Will retrain...")
        return None

    return data
