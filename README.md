# Robustness or Artifact?

[![Full-stack CI](https://github.com/mutwassim/Robustness-or-Artifact/actions/workflows/ci.yml/badge.svg)](https://github.com/mutwassim/Robustness-or-Artifact/actions/workflows/ci.yml)
[![Deploy to GitHub Pages](https://github.com/mutwassim/Robustness-or-Artifact/actions/workflows/pages.yml/badge.svg)](https://github.com/mutwassim/Robustness-or-Artifact/actions/workflows/pages.yml)

**Re-examining adversarial URL perturbation results on the PhiUSIIL phishing dataset, with a full-stack demo of the reproduced model.**

We reproduce a 2026 phishing URL detection study and show its results come from a dataset flaw: every safe URL in PhiUSIIL is a bare `https://www` homepage. A one-line rule scores 99.6%, matching trained models; real URL-only accuracy is about 86%, and the reported attack robustness hides real weaknesses. We propose fairer ways to test phishing detectors.

This repository also packages the reproduced **character TF-IDF + Logistic Regression** model as a web application: a Flask REST API, an Angular interface, automated tests with GitHub Actions, and a static demo on GitHub Pages.

**Live static demo:** https://mutwassim.github.io/Robustness-or-Artifact/

> **Status:** Work in progress. Reproduction, core analysis and the web application are complete; statistical validation and extension to more models are ongoing.

---

## Contents

- [Background](#background)
- [Key findings](#key-findings)
- [Reproduction of the original paper](#reproduction-of-the-original-paper)
- [Method](#method)
- [Web application](#web-application)
- [Run locally](#run-locally)
- [Tests and CI/CD](#tests-and-cicd)
- [API reference](#api-reference)
- [Repository structure](#repository-structure)
- [Roadmap](#roadmap)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [Authors](#authors)

---

## Background

This project starts from the following paper:

> Ahamed, T., Kakon, S. C., Al Farid, F., Uddin, J., & Abdul Karim, H. B. (2026).
> *An integrated evaluation protocol for adversarial robustness, generalization, and explanation stability in URL-based phishing detection.*
> Frontiers in Computer Science, 8, 1834407. https://doi.org/10.3389/fcomp.2026.1834407

The paper trains phishing URL classifiers on the **PhiUSIIL** dataset, reports 99.6–99.8% accuracy, and shows that the models collapse under simple URL changes such as adding a `login.` subdomain.

**Dataset:** PhiUSIIL Phishing URL Dataset (Prasad & Chandra, 2024), UCI Machine Learning Repository, id 967, licensed CC BY 4.0.
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

---

## Key findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | The benign class is a single template | 100% of safe URLs match `https://www.domain.tld` with no path, vs 1.03% of phishing URLs |
| 2 | A trivial rule matches the published models | Regex rule: **0.9958** accuracy, vs Logistic Regression 0.9957 and XGBoost 0.9972 |
| 3 | Models mostly learned the template | The original model catches only **29.7%** of phishing URLs that look like the template |
| 4 | Real URL-only signal is much weaker | Template-neutral model: **0.8647** accuracy, 0.9159 AUROC |
| 5 | Apparent robustness is blindness to lookalikes | Under TLD-swap and typo attacks, lookalike domains of safe sites are accepted as safe 100% of the time |
| 6 | The subdomain collapse is a false-alarm cascade | Phishing recall stays at 1.00 while safe-URL accuracy drops to 0.0003; real subdomains appear in 2.88% of safe vs 54.34% of phishing URLs |
| 7 | A real evasion attack is hidden | Moving phishing domains to `.com/.net/.org/.co` lowers template-neutral phishing recall from 0.7254 to 0.6142 |
| 8 | Typosquatting goes undetected | Typo variants of legitimate domains are accepted as safe by both models |

All numbers come from our own experiments with Logistic Regression on a host-disjoint test split (n = 34,354). Confidence intervals and results for more models are in progress.

---

## Reproduction of the original paper

| Model | Our test accuracy | Paper (95% CI) |
|-------|-------------------|----------------|
| XGBoost | 0.9972 | 0.9974 [0.9968, 0.9979] |
| Logistic Regression | 0.9957 | 0.9962 [0.9956, 0.9968] |

Our subdomain-attack results (0.469 / 0.414 / 0.410 at depths 1–3) also reproduce the paper's reported collapse (0.490 / 0.436 / 0.432).

---

## Method

- **Data:** PhiUSIIL URLs only. Labels inverted so that `1 = phishing`. URLs normalised (NFKC, control-character removal, lowercased scheme and host, IDNA encoding) and deduplicated, giving 235,358 URLs.
- **Split:** host-disjoint 70 / 15 / 15 using `GroupShuffleSplit` (seed 1337): 164,901 / 36,103 / 34,354.
- **Features:** character 3–5-gram TF-IDF (`min_df = 3`).
- **Models:** Logistic Regression (L2, C = 1.0) and XGBoost (300 trees, depth 6, learning rate 0.1), matching the paper.
- **Template-neutral input ("host-only"):** keeps only the host and removes every `www` label, so safe and phishing URLs share the same form.
- **Attacks:** subdomain injection, typo, TLD swap and homoglyph at depths 1–3, implemented from the paper's description. Padding and path attacks cannot be evaluated fairly on PhiUSIIL, because no safe URL has a path.
- **Metrics:** accuracy, F1 and AUROC, plus per-class phishing recall (evasion) and benign accuracy (false alarms).

---

## Web application

The deployed model is the **original-URL Logistic Regression** from the analysis, retrained by `backend/train_model.py` with exactly the notebook's preprocessing and split. It reproduces the notebook's test results exactly.

| Item | Value |
| --- | --- |
| Input | 1 field: `URL` (string, max 2,048 characters) |
| Pipeline | URL normalisation → char 3–5-gram TF-IDF → Logistic Regression, saved as one scikit-learn `Pipeline` |
| Artifact | `backend/models/model_bundle.joblib` (6.2 MB) |
| Output | `Legitimate` / `Phishing` + phishing probability |
| Test results | Accuracy 0.9957 · F1 0.9947 · AUROC 0.9988 |

### Architecture

```
Browser (Angular form, http://localhost:4200)
      │  POST /api/predict  {"URL": "..."}
      ▼
Angular dev proxy  ──►  Flask API (http://127.0.0.1:5000)
                         │  validate → normalise → TF-IDF → Logistic Regression
                         ▼
                   JSON prediction  ──►  Angular result panel

Git push ──► GitHub Actions: backend tests + Angular build
         └─► GitHub Pages: static Angular interface only
```

> **Deployment boundary:** GitHub Pages hosts the Angular interface only. It cannot run Python or the model, so on the public site the prediction button is disabled with a notice. Live predictions run only when Flask and Angular are started locally.

---

## Run locally

### Requirements

- Python 3.11, Node.js 26, npm 12, Git
- Commands below are for Windows PowerShell

### One-time setup (project root)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
cd frontend; npm install; cd ..
npm install -g @angular/cli
```

The trained model bundle is already in the repository. To retrain it from scratch:

```powershell
python -m pip install ucimlrepo
python backend\train_model.py
```

### Start the app (two terminals)

**Terminal 1 – Flask API (project root):**
```powershell
.\venv\Scripts\Activate.ps1
python backend\app.py
```

**Terminal 2 – Angular (frontend folder):**
```powershell
cd frontend
ng serve --open --proxy-config proxy.conf.json
```

Open http://localhost:4200, paste a URL or click an example, and select **Run local prediction**.

### Running the research analysis

Open `phishing_xgboost_bert.ipynb` (Google Colab recommended, GPU for BERT) and run all cells. Requirements: Python 3.9+, about 8 GB RAM, and:

```bash
pip install ucimlrepo xgboost scikit-learn pandas tldextract transformers accelerate
```

Quick check of the template rule:

```python
import pandas as pd
from ucimlrepo import fetch_ucirepo

data = fetch_ucirepo(id=967)
df = data.data.features[['URL']].copy()
df['label'] = 1 - data.data.targets['label'].values   # 1 = phishing

tmpl = df['URL'].str.match(r'^https://www\.[^/?#]+/?$')
print(pd.crosstab(df['label'], tmpl, normalize='index'))
```

The benign row (label 0) shows about 100% in the `True` column.

---

## Tests and CI/CD

```powershell
python -m pytest -q backend\tests      # 7 automated tests (no server needed)
python backend\test_pipeline.py        # direct model inference from the saved bundle
python backend\test_api.py             # live API request (Flask must be running)
```

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `ci.yml` – Full-stack CI | Push / pull request to `main`, manual | Python 3.11: install pinned requirements, run all backend tests. Node 26: `npm ci`, Angular build, upload build artifact |
| `pages.yml` – Deploy to GitHub Pages | Push to `main`, manual | Build Angular with `--base-href /Robustness-or-Artifact/` and deploy to GitHub Pages |

---

## API reference

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| GET | `/health` | none | `status`, `features`, `task`, `model_version` |
| POST | `/predict` | `{"URL": "https://..."}` | `prediction`, `is_phishing`, `phishing_probability`, `probabilities` [Legitimate, Phishing], `disclaimer` |

Invalid input (not JSON, missing or extra keys, empty URL, URL over 2,048 characters) returns **HTTP 400** with an error message.

Example:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict `
  -ContentType "application/json" -Body '{"URL":"http://login-apple.com/verify"}'
```

---

## Repository structure

```
.
├── backend/
│   ├── app.py                  # Flask API (/health, /predict)
│   ├── url_utils.py            # URL normalisation used in training and inference
│   ├── train_model.py          # Retrains and saves the model bundle
│   ├── make_payload.py         # Builds test_payload.json from sample_inputs.csv
│   ├── test_pipeline.py        # Direct inference check
│   ├── test_api.py             # Live API check
│   ├── test_payload.json
│   ├── requirements.txt
│   ├── models/model_bundle.joblib
│   └── tests/                  # pytest suite (API + structure)
├── frontend/                   # Angular app (URL form, prediction service, dev proxy)
├── data/sample_inputs.csv      # Non-sensitive example URLs
├── docs/                       # Environment record and model inventory
├── .github/workflows/          # ci.yml, pages.yml
├── phishing_xgboost_bert.ipynb # Research notebook
├── Robustness_or_Artifact_Research_Proposal.docx
└── README.md
```

The full dataset is downloaded with `ucimlrepo` and is not stored in the repository.

---

## Roadmap

- [x] Reproduce the paper's clean results and subdomain collapse
- [x] Audit benign-class structure and build the rule baseline
- [x] Template-neutral evaluation and hard-phishing subset
- [x] Per-class attack analysis (Logistic Regression)
- [x] Package the model as a Flask API with an Angular interface, CI and GitHub Pages
- [ ] Per-class analysis for XGBoost (and BERT if GPU time allows)
- [ ] Bootstrap confidence intervals, McNemar tests and multiple seeds
- [ ] Registered-domain (eTLD+1) disjoint split check
- [ ] Debiased benign URL set with realistic paths and subdomains
- [ ] Re-run all six attack families on the debiased set
- [ ] Split the analysis into separate notebooks
- [ ] Paper, preprint and submission

---

## Limitations

**Research**
- The host-only input discards path information, so its results are a lower bound, not the best achievable accuracy.
- The hard-phishing subset is small (n = 145).
- For TLD-swap, typo and homoglyph attacks, a "benign" label on a modified safe URL is not well defined; we read those results as acceptance of lookalike domains.
- Attack operators are our own implementation of the paper's description.
- Current results use one split and one model; validation is in progress.

**Web application**
- The demo is a **research and teaching prototype, not a security tool**. The deployed model inherits the template bias described above: it misses about 70% of phishing URLs that look like a bare `https://www.` homepage. Do not use it to make real security decisions.
- The Flask server is a development server bound to localhost; it is not hardened for public deployment.
- The GitHub Pages site is static and cannot make predictions.

---

## Acknowledgements

The benign-template pattern in PhiUSIIL was first noted in independent public code audits, including
[phish-drift](https://github.com/awesomedudeworld13/phish-drift),
[phishing-url-detection](https://github.com/ollivianguy/phishing-url-detection),
[phishguard](https://github.com/yuvraj1624/phishguard) and
[phishing-url-classifier](https://github.com/Panshul-mishra/phishing-url-classifier).
Our work independently verifies and quantifies it and examines its effect on published robustness results.

We thank the authors of the original paper and of the PhiUSIIL dataset for making their work publicly available.

---

## Citation

If you use this work, please cite the original paper and dataset:

```bibtex
@article{ahamed2026integrated,
  title   = {An integrated evaluation protocol for adversarial robustness, generalization, and explanation stability in URL-based phishing detection},
  author  = {Ahamed, Tanvir and Kakon, Shawon Chakrabarty and Al Farid, Fahmid and Uddin, Jia and Abdul Karim, Hezerul Bin},
  journal = {Frontiers in Computer Science},
  volume  = {8},
  pages   = {1834407},
  year    = {2026},
  doi     = {10.3389/fcomp.2026.1834407}
}

@article{prasad2024phiusiil,
  title   = {PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning},
  author  = {Prasad, Arvind and Chandra, Shalini},
  journal = {Computers \& Security},
  pages   = {103545},
  year    = {2024},
  doi     = {10.1016/j.cose.2023.103545}
}
```

A citation for this project will be added once the paper is available.

---

## Authors

- Mutwassim Haider — Air University Islamabad / Creative Technologies
- Supervisor: Sir Ali Javid Peerzada