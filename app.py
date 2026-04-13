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
import pandas as pd
from flask import Flask, request, jsonify, render_template

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH = "model.pkl"

# Expected feature columns (same order used during training)
RAW_INPUT_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = RAW_INPUT_COLUMNS

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
training_info = bundle.get("training_info", {})
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
    required = RAW_INPUT_COLUMNS
    missing  = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    # Extract raw values
    time_val   = float(data["Time"])
    amount_val = float(data["Amount"])
    v_vals     = [float(data[f"V{i}"]) for i in range(1, 29)]

    # Build a one-row DataFrame in the exact feature column order
    raw_df = pd.DataFrame([[time_val, amount_val] + v_vals], columns=RAW_INPUT_COLUMNS)

    # Transform this raw row directly through our trained total scaler
    scaled_row = scaler.transform(raw_df)

    return scaled_row


def classify_risk(fraud_prob: float) -> tuple[str, str]:
    """Map fraud probability to user-friendly risk/action labels."""
    if fraud_prob >= 0.8:
        return "Critical", "Block and manually review"
    if fraud_prob >= 0.5:
        return "High", "Hold for analyst review"
    if fraud_prob >= 0.2:
        return "Medium", "Approve with monitoring"
    return "Low", "Approve"


def build_dataset_insights() -> dict:
    """Build dataset-level stats for graph visualizations."""
    csv_path = os.path.join("dataset", "creditcard.csv")
    if not os.path.exists(csv_path):
        return {
            "available": False,
            "message": "Dataset file not found at dataset/creditcard.csv",
        }

    df = pd.read_csv(csv_path)
    fraud_count = int((df["Class"] == 1).sum())
    non_fraud_count = int((df["Class"] == 0).sum())

    amount_values = df["Amount"].to_numpy(dtype=float)
    # Clip to p99 to avoid a long-tail compressing the chart readability.
    upper = float(np.quantile(amount_values, 0.99))
    clipped_amount = np.clip(amount_values, 0, upper)
    hist_counts, hist_edges = np.histogram(clipped_amount, bins=12)
    amount_bins = [f"{hist_edges[i]:.0f}-{hist_edges[i + 1]:.0f}" for i in range(len(hist_edges) - 1)]

    hours = ((df["Time"] // 3600) % 24).astype(int)
    hour_labels = [f"{h:02d}:00" for h in range(24)]
    fraud_by_hour = []
    for h in range(24):
        hour_mask = hours == h
        total = int(hour_mask.sum())
        fraud = int(((df["Class"] == 1) & hour_mask).sum())
        rate = round((fraud / total) * 100, 3) if total else 0.0
        fraud_by_hour.append(rate)

    return {
        "available": True,
        "class_distribution": {
            "labels": ["Legitimate", "Fraud"],
            "values": [non_fraud_count, fraud_count],
        },
        "amount_histogram": {
            "labels": amount_bins,
            "values": hist_counts.tolist(),
        },
        "hourly_fraud_rate": {
            "labels": hour_labels,
            "values": fraud_by_hour,
        },
    }


dataset_insights = build_dataset_insights()


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

        risk_level, recommended_action = classify_risk(fraud_prob)

        return jsonify({
            "prediction": label,
            "probability": round(fraud_prob * 100, 2),  # as percentage
            "risk_level": risk_level,
            "recommended_action": recommended_action,
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except Exception as exc:
        return jsonify({"error": f"Internal server error: {str(exc)}"}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    POST /predict_batch
    multipart/form-data with file=<csv>
    CSV must contain Time, Amount, V1..V28 columns.
    """
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "Please upload a CSV file using form field 'file'."}), 400

    try:
        batch_df = pd.read_csv(file)
    except Exception:
        return jsonify({"error": "Unable to read CSV file. Ensure it is a valid .csv."}), 400

    if batch_df.empty:
        return jsonify({"error": "Uploaded CSV is empty."}), 400

    missing_columns = [c for c in RAW_INPUT_COLUMNS if c not in batch_df.columns]
    if missing_columns:
        return jsonify({"error": f"Missing required columns: {missing_columns}"}), 422

    features_df = batch_df[RAW_INPUT_COLUMNS].copy()
    for col in RAW_INPUT_COLUMNS:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

    invalid_rows = features_df.isna().any(axis=1)
    if invalid_rows.any():
        bad_indexes = features_df.index[invalid_rows].tolist()[:10]
        return jsonify({
            "error": "Some rows contain invalid numeric values.",
            "invalid_row_indexes": bad_indexes,
        }), 422

    raw_matrix = features_df.to_numpy(dtype=float)
    scaled_matrix = scaler.transform(raw_matrix)

    pred_classes = model.predict(scaled_matrix)
    pred_probs = model.predict_proba(scaled_matrix)[:, 1]

    result_df = features_df.copy()
    result_df["prediction"] = np.where(pred_classes == 1, "Fraud", "Not Fraud")
    result_df["fraud_probability"] = np.round(pred_probs * 100, 2)

    risk_levels = []
    recommended_actions = []
    for p in pred_probs:
        risk, action = classify_risk(float(p))
        risk_levels.append(risk)
        recommended_actions.append(action)
    result_df["risk_level"] = risk_levels
    result_df["recommended_action"] = recommended_actions

    total = int(len(result_df))
    fraud_count = int((result_df["prediction"] == "Fraud").sum())
    risk_distribution = {
        "Critical": int((result_df["risk_level"] == "Critical").sum()),
        "High": int((result_df["risk_level"] == "High").sum()),
        "Medium": int((result_df["risk_level"] == "Medium").sum()),
        "Low": int((result_df["risk_level"] == "Low").sum()),
    }

    preview_cols = ["prediction", "fraud_probability", "risk_level", "recommended_action", "Amount", "Time"]
    preview = (
        result_df[preview_cols]
        .sort_values(by="fraud_probability", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    return jsonify({
        "summary": {
            "total_rows": total,
            "fraud_count": fraud_count,
            "fraud_rate": round((fraud_count / total) * 100, 2) if total else 0.0,
            "average_fraud_probability": round(float(result_df["fraud_probability"].mean()), 2),
            "risk_distribution": risk_distribution,
        },
        "top_risky_rows": preview,
    })


@app.route("/insights", methods=["GET"])
def insights():
    """Expose model and dataset stats for UI charts."""
    response = {
        "dataset": dataset_insights,
        "training_info": training_info,
    }
    return jsonify(response)


@app.route("/health", methods=["GET"])
def health():
    """Simple health endpoint useful for deployments and checks."""
    return jsonify({"status": "ok", "model_loaded": True})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Starting Flask development server on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
