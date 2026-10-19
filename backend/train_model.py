from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from url_utils import normalize_url, normalize_frame

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "backend" / "models" / "model_bundle.joblib"

# 1. Load and flip labels (1 = phishing), exactly as in the notebook
print("Downloading PhiUSIIL dataset (UCI id 967)...")
data = fetch_ucirepo(id=967)
df = data.data.features[["URL"]].copy()
df["label"] = 1 - data.data.targets["label"].values
df["URL"] = df["URL"].map(normalize_url)
df = df.drop_duplicates(subset="URL").reset_index(drop=True)
df["host"] = df["URL"].map(lambda u: urlsplit(u).hostname or "")
print("Rows:", len(df), "| Phishing share:", round(df["label"].mean(), 4))

# 2. Same host-disjoint split as the notebook
gss = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=1337)
train_idx, rest_idx = next(gss.split(df, groups=df["host"]))
rest = df.iloc[rest_idx]
gss = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=1337)
val_i, test_i = next(gss.split(rest, groups=rest["host"]))
train, val, test = df.iloc[train_idx], rest.iloc[val_i], rest.iloc[test_i]
print("Split sizes (train, val, test):", len(train), len(val), len(test))

# 3. One pipeline: clean URL -> char TF-IDF -> Logistic Regression
pipeline = Pipeline([
    ("normalize", FunctionTransformer(normalize_frame)),
    ("tfidf", ColumnTransformer([
        ("url", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3), "URL"),
    ])),
    ("model", LogisticRegression(C=1.0, max_iter=1000)),
])

print("Training (this can take a few minutes)...")
pipeline.fit(train[["URL"]], train["label"])

pred = pipeline.predict(test[["URL"]])
proba = pipeline.predict_proba(test[["URL"]])[:, 1]
metrics = {
    "accuracy": round(accuracy_score(test["label"], pred), 4),
    "f1": round(f1_score(test["label"], pred), 4),
    "auroc": round(roc_auc_score(test["label"], proba), 4),
    "train_rows": len(train),
    "test_rows": len(test),
}
print("Test metrics:", metrics)
print("Notebook reported: accuracy 0.9957, f1 0.9947, auroc 0.9988")

# 4. Save the bundle
bundle = {
    "pipeline": pipeline,
    "features": ["URL"],
    "task": "classification",
    "class_names": {0: "Legitimate", 1: "Phishing"},
    "model_version": "1.0",
    "metrics": metrics,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, OUT, compress=3)
print(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")

# 5. Known-good checks
for url in ["https://www.southbankmosaics.com", "http://login-apple.com/verify"]:
    p = pipeline.predict_proba(pd.DataFrame({"URL": [url]}))[0, 1]
    print(f"{url} -> {'Phishing' if p >= 0.5 else 'Legitimate'} (phishing prob {p:.4f})")