import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────
#  KERNEL FUNCTION  (shown manually for clarity)
# ──────────────────────────────────────────────────

def rbf_kernel(x1, x2, gamma):
    diff = x1 - x2
    squared_distance = np.dot(diff, diff)   # same as ||x1 - x2||^2
    return float(np.exp(-gamma * squared_distance))


# ──────────────────────────────────────────────────
#  MANUAL DECISION SCORE  (for demonstration)
# ──────────────────────────────────────────────────

def svm_decision_score(trained_model, X_input):
    """
    Manually computes the SVM decision score for one sample.

    This replicates what sklearn does internally:
        f(x) = b + sum_over_support_vectors( alpha_i * y_i * K(sv_i, x) )

    A positive score → Rain prediction
    A negative score → No Rain prediction

    Note: sklearn does this much faster internally — this manual version
    is written out step-by-step to make the maths visible.
    """
    support_vectors = trained_model.support_vectors_    # the critical training points
    dual_coefficients = trained_model.dual_coef_[0]     # alpha_i * y_i for each SV
    bias = trained_model.intercept_[0]                  # bias term b
    gamma = trained_model._gamma                        # kernel width parameter

    score = bias
    for i, sv in enumerate(support_vectors):
        k = rbf_kernel(X_input[0], sv, gamma)
        score += dual_coefficients[i] * k

    return float(score)


# ──────────────────────────────────────────────────
#  TRAINING
# ──────────────────────────────────────────────────

def train_svm(X_train_scaled, y_train):
    model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)
    return model
