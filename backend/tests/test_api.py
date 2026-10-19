import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app import app, FEATURES

PAYLOAD = json.loads((ROOT / "backend/test_payload.json").read_text(encoding="utf-8"))


def test_health():
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json()["features"] == len(FEATURES)


def test_predict_valid():
    response = app.test_client().post("/predict", json=PAYLOAD)
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert body["prediction"] == "Legitimate"
    assert body["is_phishing"] is False


def test_predict_phishing_url():
    response = app.test_client().post(
        "/predict", json={"URL": "http://login-apple.com/verify"})
    assert response.status_code == 200
    assert response.get_json()["prediction"] == "Phishing"


def test_predict_missing_feature():
    incomplete = dict(PAYLOAD)
    incomplete.pop(FEATURES[0])
    response = app.test_client().post("/predict", json=incomplete)
    assert response.status_code == 400


def test_predict_empty_url():
    response = app.test_client().post("/predict", json={"URL": "   "})
    assert response.status_code == 400


def test_predict_invalid_json():
    response = app.test_client().post("/predict", data="not-json",
                                      content_type="application/json")
    assert response.status_code == 400