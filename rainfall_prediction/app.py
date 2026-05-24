import os
import warnings
import numpy as np
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore")

app = Flask(__name__)

# from ml.trainer import run_training, load_trained_models
# # from rainfall-prediction.ml.trainer import run_training, load_trained_models
# from ml.data_loader import FEATURE_NAMES, FEATURE_LABELS
from rainfall_prediction.ml.trainer import run_training, load_trained_models
from rainfall_prediction.ml.data_loader import FEATURE_NAMES, FEATURE_LABELS

print("Starting up RainSense...")

saved = load_trained_models()
if saved is not None:
    print("Loaded existing models from models/trained_models.pkl")
    MODELS      = saved["models"]
    SCALER      = saved["scaler"]
    METRICS     = saved["metrics"]
    DIAGNOSTICS = saved["diagnostics"]
else:
    print("No saved models found. Training from scratch...")
    MODELS, SCALER, METRICS, DIAGNOSTICS = run_training(csv_path="rainfall_prediction/Rainfall.csv")

@app.route("/")
def index():
    meta = DIAGNOSTICS.get("_meta", {})
    cb   = meta.get("class_balance", {})
    return render_template("index.html", metrics=METRICS, meta=cb)


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/diagnostics")
def api_diagnostics():
    return jsonify(DIAGNOSTICS)


@app.route("/api/feature-importance")
def feature_importance():
    rf = MODELS["Random Forest"]
    importances = rf.feature_importances_.tolist()
    pairs = sorted(
        zip(FEATURE_NAMES, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    return jsonify({
        "features":    [FEATURE_LABELS[p[0]] for p in pairs],
        "importances": [round(p[1] * 100, 2)  for p in pairs],
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True)
        raw_features = np.array(
            [float(body[k]) for k in FEATURE_NAMES], dtype=float
        ).reshape(1, -1)
        scaled_features = SCALER.transform(raw_features)
        predictions = {}
        for model_name, model in MODELS.items():
            input_for_model = scaled_features if model_name == "SVM" else raw_features

            predicted_class = int(model.predict(input_for_model)[0])
            probabilities   = model.predict_proba(input_for_model)[0]
            rain_prob       = round(float(probabilities[1]) * 100, 1)

            predictions[model_name] = {
                "prediction":  "Yes" if predicted_class == 1 else "No",
                "probability": rain_prob,
            }

        return jsonify({"success": True, "predictions": predictions})

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
