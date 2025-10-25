# train_models.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score
from bs4 import BeautifulSoup
import joblib

# Path to your prepared dataset
DATA_CSV = "data/combined_spam.csv"

# ---------- 1. Load dataset ----------
print("📂 Loading dataset...")
df = pd.read_csv(DATA_CSV)
print(f"✅ Loaded {len(df)} emails.")

# ---------- 2. Basic cleaning ----------
print("🧹 Cleaning text...")

def clean_html(text):
    try:
        return BeautifulSoup(str(text), "html5lib").get_text(separator=" ")
    except Exception:
        return str(text)

df["subject"] = df["subject"].fillna("")
df["text"] = df["text"].fillna("")
df["text_all"] = (df["subject"] + " " + df["text"]).apply(clean_html).str.lower()

# ---------- 3. Split data ----------
X = df["text_all"].values
y = (df["label"].str.lower() == "spam").astype(int).values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 4. TF-IDF Vectorizer ----------
print("🔠 Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    stop_words="english"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

# ---------- 5. Train models ----------
print("🤖 Training models...")

# Logistic Regression
lr = LogisticRegression(max_iter=300, class_weight="balanced")
lr.fit(X_train_tfidf, y_train)
joblib.dump(lr, "models/lr_model.joblib")

# SVM (linear kernel, with probability=True)
svm = SVC(kernel="linear", probability=True, class_weight="balanced")
svm.fit(X_train_tfidf, y_train)
joblib.dump(svm, "models/svm_model.joblib")

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
joblib.dump(nb, "models/nb_model.joblib")

# ---------- 6. Evaluate ----------
print("\n📊 Model evaluation on test data:")
for name, model in [("Logistic Regression", lr), ("SVM", svm), ("Naive Bayes", nb)]:
    y_pred_prob = model.predict_proba(X_test_tfidf)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    print(f"\n🔹 {name} | AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

print("\n✅ All models trained and saved in /models folder!")
