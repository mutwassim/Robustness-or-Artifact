from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
bundle = joblib.load(ROOT / "backend/models/model_bundle.joblib")
with (ROOT / "backend/test_payload.json").open(encoding="utf-8") as f:
    payload = json.load(f)
features = bundle["features"]
assert set(features) == set(payload), "Payload keys must match features"
frame = pd.DataFrame([[payload[f] for f in features]], columns=features)
model = bundle["pipeline"]
pred = model.predict(frame).tolist()
print("Input:", payload)
print("Prediction:", pred, "->", bundle["class_names"][pred[0]])
print("Probabilities [Legitimate, Phishing]:", model.predict_proba(frame).round(4).tolist())
print("PASS: local inference")