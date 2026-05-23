"""
random_forest_model.py
----------------------
Random Forest model for rainfall prediction.

How Random Forest works:
  A Random Forest is an ensemble of decision trees. Each tree is trained on:
    (a) a random bootstrap sample of the training data (sampling with replacement)
    (b) a random subset of features at each split

  For a final prediction, all trees vote and the majority wins (for classification).
  For probabilities, we average the probability outputs across all trees.

  This "wisdom of the crowd" approach reduces overfitting that a single
  tree would have — a tree that memorised noise is outvoted by others.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


# ──────────────────────────────────────────────────
#  VOTE AGGREGATION  (shown manually for clarity)
# ──────────────────────────────────────────────────

def majority_vote(predictions_list):
    """
    Aggregates binary predictions from multiple trees via majority vote.

    Each tree outputs 0 (No Rain) or 1 (Rain).
    If more than half vote Rain → final answer is Rain.

    This is exactly how RandomForestClassifier.predict() works internally,
    just written out explicitly here for clarity.

    Example:
        trees predict: [1, 0, 1, 1, 0]  →  3 Rain, 2 No Rain  →  Rain
    """
    predictions = np.array(predictions_list)
    rain_votes   = int(predictions.sum())
    total_trees  = len(predictions)
    return 1 if rain_votes > total_trees / 2 else 0


def average_probabilities(prob_list):
    """
    Averages rain probabilities from multiple trees.

    Each tree outputs a probability (0.0 to 1.0) for Rain.
    The forest's probability is simply the mean across all trees.

    This is what RandomForestClassifier.predict_proba() returns — the
    average of each tree's predict_proba() outputs.

    Example:
        tree probs: [0.8, 0.45, 0.72, 0.61, 0.53]
        forest prob = (0.8 + 0.45 + 0.72 + 0.61 + 0.53) / 5 = 0.622
    """
    probs = np.array(prob_list)
    return float(probs.mean())


def show_tree_agreement(trained_model, X_input, n_show=10):
    """
    Returns per-tree predictions for one input to show how trees "vote".
    Shows only the first n_show trees to keep output readable.

    This makes the ensemble aggregation visible:
        Tree 0: Rain (prob 0.76)
        Tree 1: No Rain (prob 0.38)
        ...
    """
    results = []
    trees_to_check = trained_model.estimators_[:n_show]

    for i, tree in enumerate(trees_to_check):
        proba = tree.predict_proba(X_input)[0]
        rain_prob = float(proba[1]) if len(proba) > 1 else 0.0
        pred = "Rain" if rain_prob >= 0.5 else "No Rain"
        results.append(f"  Tree {i:02d}: {pred}  (rain prob = {rain_prob:.3f})")

    return results


# ──────────────────────────────────────────────────
#  TRAINING
# ──────────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest with 200 trees.

    Hyperparameters chosen:
        n_estimators=200   - 200 trees (more trees = more stable, diminishing returns past ~150)
        max_depth=8        - each tree can be at most 8 levels deep
        min_samples_leaf=4 - leaf must have at least 4 samples (prevents tiny leaves)
        n_jobs=-1          - use all CPU cores for parallel training
        class_weight=balanced - weights inversely proportional to class frequency
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model
