from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE = Path(__file__).resolve().parent
BUNDLE = joblib.load(BASE / "models/model_bundle.joblib")
MODEL = BUNDLE["pipeline"]
FEATURES = BUNDLE["features"]          # ["URL"]
TASK = BUNDLE.get("task", "classification")
CLASS_NAMES = BUNDLE["class_names"]    # {0: "Legitimate", 1: "Phishing"}
MAX_URL_LENGTH = 2048

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", features=len(FEATURES), task=TASK,
                   model_version=BUNDLE.get("model_version"))


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object"), 400

    missing = [f for f in FEATURES if f not in data]
    extra = [f for f in data if f not in FEATURES]
    if missing or extra:
        return jsonify(error="Input schema mismatch",
                       missing=missing, extra=extra), 400

    url = data["URL"]
    if not isinstance(url, str) or not url.strip():
        return jsonify(error="URL must be a non-empty string"), 400
    if len(url) > MAX_URL_LENGTH:
        return jsonify(error=f"URL longer than {MAX_URL_LENGTH} characters"), 400

    frame = pd.DataFrame([[url]], columns=FEATURES)
    try:
        proba = MODEL.predict_proba(frame)[0]
        label = int(proba.argmax())
        return jsonify(
            prediction=CLASS_NAMES[label],
            is_phishing=bool(label == 1),
            phishing_probability=round(float(proba[1]), 4),
            probabilities=[round(float(p), 4) for p in proba],
            task=TASK,
            model_version=BUNDLE.get("model_version"),
            disclaimer=("Research/teaching prototype only. This model relies "
                        "heavily on URL templates and misses many phishing URLs "
                        "that look like normal homepages. Do not use it to make "
                        "real security decisions."),
        )
    except (ValueError, TypeError) as exc:
        return jsonify(error="Invalid input", detail=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction processing failed"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)