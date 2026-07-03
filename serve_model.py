import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model and scaler once when the server starts
# (not on every request — loading is slow, prediction is fast)
MODEL_PATH = "model/isolation_forest.pkl"
SCALER_PATH = "model/scaler.pkl"

print("Loading model...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model loaded successfully.")

# ── Health check endpoint ─────────────────────────────────────────────────────
# SageMaker calls this to verify the container is alive
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "healthy"}), 200

# ── Prediction endpoint ───────────────────────────────────────────────────────
# SageMaker sends data here and expects predictions back
# Matches SageMaker's /invocations endpoint convention exactly
@app.route("/invocations", methods=["POST"])
def predict():
    try:
        # Parse the incoming JSON request
        data = request.get_json()

        # Expect a list of data points, each with 5 features:
        # [close, rolling_avg_close, volatility, pct_change, volume_zscore]
        instances = data.get("instances", [])

        if not instances:
            return jsonify({"error": "No instances provided"}), 400

        # Convert to numpy array and scale
        X = np.array(instances)
        X_scaled = scaler.transform(X)

        # Run prediction
        # -1 = anomaly, 1 = normal
        predictions = model.predict(X_scaled).tolist()
        scores = model.score_samples(X_scaled).tolist()

        # Build readable response
        results = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            results.append({
                "instance": i,
                "prediction": "anomaly" if pred == -1 else "normal",
                "anomaly_score": round(score, 4),
                "raw_prediction": pred
            })

        return jsonify({
            "predictions": results,
            "total_instances": len(results),
            "anomalies_detected": sum(1 for r in results if r["prediction"] == "anomaly")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Run the server ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting prediction server on port {port}...")
    print(f"Health check: http://localhost:{port}/ping")
    print(f"Predictions:  http://localhost:{port}/invocations")
    app.run(host="0.0.0.0", port=port, debug=False)