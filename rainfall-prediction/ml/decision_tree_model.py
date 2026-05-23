"""
decision_tree_model.py
----------------------
Decision Tree model for rainfall prediction.

How a Decision Tree works:
  The tree recursively splits the dataset on the feature that produces
  the biggest reduction in "impurity" (how mixed rain/no-rain samples are).

  We measure impurity with Gini index:
      Gini(S) = 1 - sum_k( p_k^2 )
  where p_k is the proportion of class k in sample set S.

  At each node, the split that maximises Information Gain is chosen:
      IG = Gini(parent) - weighted_avg( Gini(left), Gini(right) )

  The tree grows until max_depth is reached or nodes are too small.
  For probability output, we wrap it in CalibratedClassifierCV which
  smooths the leaf-level probabilities (raw DT leaves output 0 or 1,
  which is too extreme).
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV


# ──────────────────────────────────────────────────
#  GINI IMPURITY  (shown manually for clarity)
# ──────────────────────────────────────────────────

def gini_impurity(labels):
    """
    Computes the Gini impurity of a set of labels.

    Formula:  Gini(S) = 1 - sum_k( p_k^2 )

    A Gini of 0 means all samples are the same class (pure node).
    A Gini of 0.5 means 50/50 split (maximally impure for binary case).
    """
    if len(labels) == 0:
        return 0.0
    total = len(labels)
    classes, counts = np.unique(labels, return_counts=True)
    gini = 1.0
    for count in counts:
        p_k = count / total
        gini -= p_k ** 2
    return float(gini)


def information_gain(parent_labels, left_labels, right_labels):
    """
    Computes the information gain of a split.

    IG = Gini(parent) - (|left|/|parent|) * Gini(left)
                      - (|right|/|parent|) * Gini(right)

    Higher IG = better split — the feature is more useful for the split.
    The Decision Tree picks the feature+threshold that maximises IG at each node.
    """
    n_parent = len(parent_labels)
    n_left   = len(left_labels)
    n_right  = len(right_labels)

    if n_parent == 0:
        return 0.0

    parent_gini = gini_impurity(parent_labels)
    left_gini   = gini_impurity(left_labels)
    right_gini  = gini_impurity(right_labels)

    weighted_child_gini = (n_left / n_parent) * left_gini \
                        + (n_right / n_parent) * right_gini

    return float(parent_gini - weighted_child_gini)


# ──────────────────────────────────────────────────
#  TRAINING
# ──────────────────────────────────────────────────

def train_decision_tree(X_train, y_train):
    """
    Trains a Decision Tree and wraps it with CalibratedClassifierCV.

    Why calibration?
      Raw decision tree leaves output 0 or 1 (whichever class dominates
      that leaf). After calibration, probabilities like 0.63, 0.41 become
      possible — much more useful for showing rain % to the user.

    Hyperparameters:
        max_depth=5         - tree can't go deeper than 5 levels
        min_samples_leaf=8  - a leaf must have at least 8 samples
        class_weight=balanced - accounts for rain/no-rain imbalance
    """
    base_tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced",
    )

    # Isotonic calibration uses a non-parametric approach (better than sigmoid
    # when we have enough data). cv=3 means 3-fold internal cross-val.
    calibrated_tree = CalibratedClassifierCV(base_tree, cv=3, method="isotonic")
    calibrated_tree.fit(X_train, y_train)

    return calibrated_tree


# ──────────────────────────────────────────────────
#  TREE TRAVERSAL (manual, for demonstration)
# ──────────────────────────────────────────────────

def trace_decision_path(trained_model, X_input, feature_names):
    """
    Manually traces the path through the underlying decision tree
    for one sample, showing which feature is tested at each node.

    This makes the tree's "if-then" logic visible.
    Returns a list of decision strings like:
        ["humidity <= 78.50 → go left", "cloud > 55.0 → go right", ...]
    """
    # CalibratedClassifierCV stores calibrated classifiers, each wrapping a base tree
    # Access one of the underlying base estimators
    try:
        base_est = trained_model.calibrated_classifiers_[0].estimator
    except AttributeError:
        try:
            base_est = trained_model.calibrated_classifiers_[0].base_estimator
        except AttributeError:
            return ["(tree trace not available for this sklearn version)"]

    tree = base_est.tree_
    node_id = 0
    path = []

    while tree.feature[node_id] >= 0:   # -2 means leaf node
        feat_idx   = tree.feature[node_id]
        threshold  = tree.threshold[node_id]
        feat_name  = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat_{feat_idx}"
        value      = float(X_input[0][feat_idx])

        if value <= threshold:
            direction = "≤ threshold → left branch"
            node_id = tree.children_left[node_id]
        else:
            direction = "> threshold → right branch"
            node_id = tree.children_right[node_id]

        path.append(f"{feat_name} = {value:.2f}  (threshold {threshold:.2f})  {direction}")

    return path
