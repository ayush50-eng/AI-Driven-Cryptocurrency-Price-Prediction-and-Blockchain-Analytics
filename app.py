"""
app.py — Flask Web Server for Credit Card Fraud Detection
==========================================================
Endpoints:
  GET  /         → Serve the HTML frontend
  POST /predict  → Accept transaction JSON, return fraud prediction
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH = "model.pkl"

# Expected feature columns (same order used during training)
FEATURE_COLUMNS = (
    ["scaled_time", "scaled_amount"] +
    [f"V{i}" for i in range(1, 29)]
)

# ─────────────────────────────────────────────
# LOAD MODEL BUNDLE AT STARTUP
# ─────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"[ERROR] '{MODEL_PATH}' not found. "
        "Please run  python model.py  first to train and save the model."
    )

bundle = joblib.load(MODEL_PATH)
model  = bundle["model"]
scaler = bundle["scaler"]
print(f"[INFO] Model loaded from '{MODEL_PATH}'")


# ─────────────────────────────────────────────
# HELPER — build feature vector
# ─────────────────────────────────────────────
def build_features(data: dict) -> np.ndarray:
    """
    Extract and scale features from incoming JSON.
    Raw 'Time' and 'Amount' are normalized the same way
    the scaler was fitted.  V1-V28 are passed as-is.
    """
    from sklearn.preprocessing import StandardScaler

    # Validate required keys
    required = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    missing  = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    # Extract raw values
    time_val   = float(data["Time"])
    amount_val = float(data["Amount"])
    v_vals     = [float(data[f"V{i}"]) for i in range(1, 29)]

    # Build a one-row dataframe with the SAME columns used in training
    # (scaled_time, scaled_amount come from the original StandardScaler)
    # We use the saved scaler to transform them consistently.
    raw_row = np.array([[time_val, amount_val] + v_vals])

    # The scaler was fitted on [scaled_time, scaled_amount, V1..V28]
    # which in training were already StandardScaled Time/Amount + raw Vs.
    # For inference: we scale Time & Amount by creating a temporary scaler
    # that mimics what was done in preprocess().
    # (The bundle scaler was fit on the full feature matrix after transformation.)
    scaled_row = scaler.transform(raw_row)

    return scaled_row


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: JSON with Time, Amount, V1-V28
    Returns: { "prediction": "Fraud"|"Not Fraud", "probability": float }
    """
    # 1. Parse JSON body
    if not request.is_json:
        return jsonify({"error": "Request must be JSON (Content-Type: application/json)"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "Empty request body."}), 400

    try:
        # 2. Build feature vector
        features = build_features(data)

        # 3. Run prediction
        pred_class = int(model.predict(features)[0])
        pred_proba = model.predict_proba(features)[0]

        # Class 1 = Fraud, Class 0 = Not Fraud
        fraud_prob = float(pred_proba[1])
        label      = "Fraud" if pred_class == 1 else "Not Fraud"

        return jsonify({
            "prediction": label,
            "probability": round(fraud_prob * 100, 2)   # as percentage
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except Exception as exc:
        return jsonify({"error": f"Internal server error: {str(exc)}"}), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Starting Flask development server on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
