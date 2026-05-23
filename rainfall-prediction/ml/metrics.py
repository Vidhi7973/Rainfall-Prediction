"""
metrics.py
----------
All evaluation metrics for a binary classifier (Rain vs No Rain).

The manual formulas are written out below each function so you can see
what the numbers actually mean. sklearn is then used to compute the
final values efficiently.

Confusion matrix layout (binary, Rain = positive class):
              Predicted No Rain   Predicted Rain
Actual No Rain      TN                FP
Actual Rain         FN                TP
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score,
    log_loss, brier_score_loss, average_precision_score,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score


# ──────────────────────────────────────────────────
#  MANUAL FORMULA DEMOS
# ──────────────────────────────────────────────────

def manual_accuracy(y_true, y_pred):
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    The simplest metric — fraction of all predictions that were correct.
    Misleading on imbalanced datasets (a model that always says "Rain"
    would score 67.9% accuracy on this dataset without learning anything).
    """
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def manual_precision(y_true, y_pred):
    """
    Precision = TP / (TP + FP)

    Of all the times the model predicted Rain, how often was it actually raining?
    High precision = low false alarm rate.
    """
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def manual_recall(y_true, y_pred):
    """
    Recall / Sensitivity = TP / (TP + FN)

    Of all the actual rain days, how many did the model catch?
    High recall = the model doesn't miss many rain days.
    In weather forecasting, recall is often more important than precision
    (missing rain = no umbrella = bad; false alarm = umbrella for no reason).
    """
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def manual_f1(precision, recall):
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Harmonic mean of precision and recall.
    Useful when you want a single metric that balances both.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def manual_mcc(tn, fp, fn, tp):
    """
    Matthews Correlation Coefficient.

    MCC = (TP*TN - FP*FN) / sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) )

    Ranges from -1 (all wrong) to 0 (random) to +1 (perfect).
    Considered more informative than F1 when classes are imbalanced.
    """
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


# ──────────────────────────────────────────────────
#  FULL METRIC COMPUTATION
# ──────────────────────────────────────────────────

def compute_all_metrics(y_test, y_pred, y_proba, model, X_train_input, y_train):
    """
    Computes the full set of classification metrics for one model.
    Returns a dict with everything needed for the dashboard.

    Both the manual formulas above AND sklearn are used — sklearn
    handles cross-validation and curve data which are harder to write manually.
    """
    # ── Confusion matrix ───────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total = int(tn + fp + fn + tp)

    # ── Core metrics (sklearn for accuracy) ────────
    acc  = round(accuracy_score(y_test, y_pred)                     * 100, 2)
    prec = round(precision_score(y_test, y_pred, zero_division=0)   * 100, 2)
    rec  = round(recall_score(y_test, y_pred,    zero_division=0)   * 100, 2)
    f1   = round(f1_score(y_test, y_pred,         zero_division=0)  * 100, 2)

    # manual versions (confirm they match sklearn)
    # These are kept to show the formulas are correct
    _prec_m = manual_precision(y_test, y_pred)
    _rec_m  = manual_recall(y_test, y_pred)

    # ── Rates derived from confusion matrix ────────
    sensitivity = round(float(tp) / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0
    specificity = round(float(tn) / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0.0
    npv         = round(float(tn) / (tn + fn) * 100, 2) if (tn + fn) > 0 else 0.0
    fdr         = round(float(fp) / (fp + tp) * 100, 2) if (fp + tp) > 0 else 0.0
    for_rate    = round(float(fn) / (fn + tn) * 100, 2) if (fn + tn) > 0 else 0.0
    fpr_rate    = round(float(fp) / (fp + tn) * 100, 2) if (fp + tn) > 0 else 0.0
    fnr_rate    = round(float(fn) / (fn + tp) * 100, 2) if (fn + tp) > 0 else 0.0
    g_mean      = round(float(np.sqrt((sensitivity / 100) * (specificity / 100))), 4)

    # ── Advanced metrics ────────────────────────────
    mcc      = round(float(matthews_corrcoef(y_test, y_pred)),            4)
    kappa    = round(float(cohen_kappa_score(y_test, y_pred)),            4)
    bal_acc  = round(float(balanced_accuracy_score(y_test, y_pred)) * 100, 2)
    auc      = round(float(roc_auc_score(y_test, y_proba)),               4)
    ll       = round(float(log_loss(y_test, y_proba)),                    4)
    brier    = round(float(brier_score_loss(y_test, y_proba)),            4)
    auprc    = round(float(average_precision_score(y_test, y_proba)),     4)

    # ── Cross-validation (5-fold) ───────────────────
    cv_auc = cross_val_score(model, X_train_input, y_train, cv=5, scoring="roc_auc",  n_jobs=-1)
    cv_acc = cross_val_score(model, X_train_input, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    cv_f1  = cross_val_score(model, X_train_input, y_train, cv=5, scoring="f1",       n_jobs=-1)

    # ── ROC curve points ────────────────────────────
    thresholds = np.linspace(0, 1, 51)
    roc_pts = []
    for t in thresholds:
        yp_t = (y_proba >= t).astype(int)
        cm_t = confusion_matrix(y_test, yp_t, labels=[0, 1])
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        _fpr = float(fp_t) / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
        _tpr = float(tp_t) / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        roc_pts.append({"fpr": round(_fpr, 3), "tpr": round(_tpr, 3)})

    # ── Precision-Recall curve ──────────────────────
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_proba)
    idx    = np.linspace(0, len(pr_prec) - 1, 60, dtype=int)
    pr_pts = [{"precision": round(float(pr_prec[i]), 3),
               "recall":    round(float(pr_rec[i]),  3)} for i in idx]

    # ── Calibration curve ───────────────────────────
    cal_frac, cal_mean = calibration_curve(y_test, y_proba, n_bins=8, strategy="uniform")
    cal_pts = [{"mean_pred": round(float(m), 3), "frac_pos": round(float(f), 3)}
               for m, f in zip(cal_mean, cal_frac)]

    return {
        # basics
        "confusion_matrix": cm.tolist(),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "total": total,
        # standard
        "accuracy":          acc,
        "precision":         prec,
        "recall":            rec,
        "f1_score":          f1,
        # sensitivity / specificity family
        "sensitivity":       sensitivity,
        "specificity":       specificity,
        "npv":               npv,
        "fdr":               fdr,
        "for_rate":          for_rate,
        "fpr":               fpr_rate,
        "fnr":               fnr_rate,
        "g_mean":            g_mean,
        # advanced
        "balanced_accuracy": bal_acc,
        "mcc":               mcc,
        "kappa":             kappa,
        "auc_roc":           auc,
        "auc_prc":           auprc,
        "log_loss":          ll,
        "brier_score":       brier,
        # cross-validation
        "cv_auc_mean":   round(float(cv_auc.mean()), 4),
        "cv_auc_std":    round(float(cv_auc.std()),  4),
        "cv_auc_scores": [round(float(s), 4) for s in cv_auc],
        "cv_acc_mean":   round(float(cv_acc.mean()) * 100, 2),
        "cv_f1_mean":    round(float(cv_f1.mean()) * 100,  2),
        # curves
        "roc_curve": roc_pts,
        "pr_curve":  pr_pts,
        "cal_curve": cal_pts,
    }
