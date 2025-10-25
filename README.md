# Spam Email Detection

## Overview
A Flask web app that classifies emails as SPAM or HAM, integrates with Gmail, and visualizes model performance. It supports adding your own emails to the dataset and retraining models.

## Features
- Gmail link & scan: Link your Gmail and check unread emails.
- Results UI: Styled results table with clear SPAM/HAM emphasis.
- Accuracy page: High-DPI charts and confusion matrices for each model and ensemble.
- Retraining pipeline: Build dataset from raw emails, train TF‑IDF + LR/SVM/NB, save to `models/`.
- Customizable thresholds: Tune per-model and ensemble thresholds.

## Tech Stack
- Python, Flask
- scikit-learn, pandas, numpy, matplotlib
- Gmail API (google-api-python-client, google-auth)

## Project Structure
- `app.py` – Flask app and routes.
- `spam_classifier.py` – Loads vectorizer/models, prediction + thresholds.
- `gmail_spam_checker.py` – Gmail auth and helpers.
- `prepare_dataset.py` – Build `data/combined_spam.csv` from raw mail dirs.
- `train_models.py` – Train TF‑IDF + LR/SVM/NB and save artifacts.
- `templates/` – HTML pages (`index.html`, `dashboard.html`, `results.html`, `accuracy_page.html`).
- `static/` – CSS and images.
- `easy_ham/`, `hard_ham/`, `spam_2/` – Raw mail directories (dataset sources).
- `models/` – Saved artifacts (`*.joblib`) used by the app.
- `data/` – Generated dataset CSVs.

## Setup

- Create and activate venv
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

- Install dependencies
```powershell
pip install -r requirements.txt
```
If `requirements.txt` is missing, install:
```powershell
pip install flask scikit-learn pandas numpy matplotlib google-api-python-client google-auth google-auth-oauthlib beautifulsoup4 html5lib joblib tqdm
```

## Gmail API Setup
- Place your `credentials.json` in project root (ignored by Git).
- First link will open browser for OAuth and create `token.pickle` (also ignored by Git).

## Build Dataset
- Put raw emails in:
  - `easy_ham/` and/or `hard_ham/` for HAM.
  - `spam_2/` for SPAM.
- Generate CSV:
```powershell
python .\prepare_dataset.py
```
- Output: `data/combined_spam.csv`

## Train Models
- Train TF‑IDF + Logistic Regression + SVM + Naive Bayes:
```powershell
python .\train_models.py
```
- Saves to `models/`:
  - `tfidf_vectorizer.joblib`
  - `lr_model.joblib`
  - `svm_model.joblib`
  - `nb_model.joblib`

## Run the App
```powershell
python .\app.py
```
- Home: `http://127.0.0.1:5000/`
- Dashboard: `/dashboard`
- Link Gmail: `/link_gmail`
- Check emails: `/check_emails`
- Metrics: `/accuracy`

## Viewing Metrics
- The `/accuracy` route:
  - Reads `data/combined_spam.csv`
  - Computes per-model predictions using `SpamClassifier` thresholds and ensemble
  - Renders high-DPI charts and confusion matrices to `templates/accuracy_page.html`

## Reducing False Positives (Important Emails Marked SPAM)
- Add such emails to `easy_ham/` or `hard_ham/` as raw `.eml` (headers + body).
- Rebuild dataset and retrain:
```powershell
python .\prepare_dataset.py
python .\train_models.py
```
- Adjust thresholds in `spam_classifier.py` if needed:
  - `DEFAULT_THRESHOLDS = {"lr": 0.75, "svm": 0.75, "nb": 0.75}`
  - `DEFAULT_ENSEMBLE_THRESHOLD = 0.60`

## GitHub (Quick)
```powershell
git init
git branch -M main
git add .
git commit -m "Initial commit: Spam Email Detection app"
git remote add origin https://github.com/<YOUR_USERNAME>/Spam_Email_Detection.git
git push -u origin main
```
- Add collaborators: GitHub → Repo → Settings → Collaborators.

## Security Notes
- Never commit `credentials.json` or `token.pickle`. They are ignored via `.gitignore`.
- `data/*.csv` and `models/*.joblib` are also ignored to keep repo lean.

## License
Add your preferred license (e.g., MIT) as `LICENSE`.

## Acknowledgements
- SpamAssassin datasets structure for `easy_ham/`, `hard_ham/`, `spam_2/`.
- scikit-learn, Flask, Google APIs.
