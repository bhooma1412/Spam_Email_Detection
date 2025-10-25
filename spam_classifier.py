# spam_classifier.py
"""
SpamClassifier: load saved TF-IDF and models, provide predict(text) that returns
per-model probabilities, votes, average probability and final is_spam decision.

Expectations:
 - Models saved in ./models: tfidf_vectorizer.joblib, lr_model.joblib, svm_model.joblib, nb_model.joblib
 - Each model should either implement predict_proba or decision_function.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

DEFAULT_MODEL_FILES = {
    "lr": "lr_model.joblib",
    "svm": "svm_model.joblib",
    "nb": "nb_model.joblib"
}

DEFAULT_THRESHOLDS = {
    "lr": 0.75,
    "svm": 0.75,
    "nb": 0.75
}
DEFAULT_ENSEMBLE_THRESHOLD = 0.60


class SpamClassifier:
    def __init__(
        self,
        model_files: Dict[str, str] = None,
        thresholds: Dict[str, float] = None,
        ensemble_threshold: float = None,
    ):
        self.model_files = model_files or DEFAULT_MODEL_FILES
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()
        self.ensemble_threshold = ensemble_threshold or DEFAULT_ENSEMBLE_THRESHOLD

        # load vectorizer
        self.vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
        # load models
        self.models = {}
        for key, fname in self.model_files.items():
            path = os.path.join(MODEL_DIR, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            self.models[key] = joblib.load(path)

    def _model_prob(self, model, X):
        """
        Return a float probability in [0,1] for a single sample X (sparse matrix)
        Handles predict_proba, decision_function, fallback to predict.
        """
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[:, 1][0])
        # some SVMs have decision_function
        if hasattr(model, "decision_function"):
            try:
                df = model.decision_function(X)[0]
                # map to (0,1)
                return float(1.0 / (1.0 + np.exp(-df)))
            except Exception:
                pass
        # fallback: predict returns 0/1
        pred = model.predict(X)[0]
        return float(pred)

    def predict_proba(self, text: str) -> Dict[str, Any]:
        """
        Returns:
        {
            'model_probs': {'lr':0.99, 'svm':0.87, 'nb':0.92},
            'avg_prob': 0.9266,
            'votes': {'lr':1,'svm':0,'nb':0},
        }
        """
        X = self.vectorizer.transform([text])
        probs = {}
        for key, model in self.models.items():
            p = self._model_prob(model, X)
            probs[key] = p
        avg = float(np.mean(list(probs.values())))
        votes = {k: int(probs[k] >= self.thresholds.get(k, 0.5)) for k in probs.keys()}
        return {"model_probs": probs, "avg_prob": avg, "votes": votes}

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Returns a dict with:
        - is_spam (bool)
        - avg_prob (float)
        - model_probs (dict)
        - votes (dict)
        - decision_reason (str)
        """
        out = self.predict_proba(text)
        votes_sum = sum(out["votes"].values())
        avg = out["avg_prob"]

        # Decision logic:
        # - If >= 2 models vote spam -> spam
        # - OR if average prob >= ensemble_threshold -> spam
        is_spam = (votes_sum >= 2) or (avg >= self.ensemble_threshold)

        if votes_sum >= 2:
            reason = f"{votes_sum} models voted spam"
        elif avg >= self.ensemble_threshold:
            reason = f"avg_prob >= ensemble_threshold ({avg:.3f} >= {self.ensemble_threshold})"
        else:
            reason = "not enough votes and avg below ensemble threshold"

        out.update({"is_spam": bool(is_spam), "decision_reason": reason})
        return out

# simple CLI test when run directly
if __name__ == "__main__":
    import argparse, textwrap
    parser = argparse.ArgumentParser(description="Test spam_classifier.py")
    parser.add_argument("--text", "-t", help="Text to classify (subject+body)", type=str)
    parser.add_argument("--file", "-f", help="Path to text file containing email body", type=str)
    args = parser.parse_args()

    if not args.text and not args.file:
        print("Provide --text or --file")
        raise SystemExit(1)

    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="ignore") as fh:
            sample = fh.read()
    else:
        sample = args.text

    clf = SpamClassifier()
    res = clf.predict(sample)
    print(textwrap.dedent(f"""
    Model probabilities: {res['model_probs']}
    Average probability: {res['avg_prob']:.4f}
    Votes: {res['votes']}
    Final decision: {'SPAM' if res['is_spam'] else 'HAM'} 
    Reason: {res['decision_reason']}
    """))
