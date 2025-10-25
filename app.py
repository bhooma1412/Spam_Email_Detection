from flask import Flask, render_template, redirect, url_for, session, flash
from gmail_spam_checker import authenticate_gmail, get_unread_messages, get_message_content
from spam_classifier import SpamClassifier
import os
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = "bhoomi_spam_email_detection_2025"

classifier = SpamClassifier()

# ------------------- HOME PAGE -------------------
@app.route("/")
def home():
    return render_template("index.html")

# ------------------- LINK GMAIL -------------------
@app.route("/link_gmail")
def link_gmail():
    try:
        # Always force user to log in again
        if os.path.exists("token.pickle"):
            os.remove("token.pickle")

        service = authenticate_gmail(force_new_login=True)
        session["gmail_linked"] = True
        flash("✅ Gmail account linked successfully!", "success")
        return redirect(url_for("dashboard"))
    except Exception as e:
        flash(f"❌ Failed to link Gmail account: {e}", "danger")
        return redirect(url_for("home"))

# ------------------- DASHBOARD -------------------
@app.route("/dashboard")
def dashboard():
    if not session.get("gmail_linked"):
        flash("Please link your Gmail account first.", "warning")
        return redirect(url_for("home"))
    return render_template("dashboard.html")

# ------------------- CHECK EMAIL STATUS -------------------
from datetime import datetime

@app.route("/check_emails")
def check_emails():
    try:
        service = authenticate_gmail(force_new_login=False)
        emails = get_unread_messages(service)
        results = []

        for msg in emails:
            msg_id = msg["id"]
            msg_detail = service.users().messages().get(
                userId='me', id=msg_id, format='metadata',
                metadataHeaders=['From', 'Date']
            ).execute()

            sender = next((h['value'] for h in msg_detail['payload']['headers'] if h['name'] == 'From'), 'Unknown')
            date_str = next((h['value'] for h in msg_detail['payload']['headers'] if h['name'] == 'Date'), None)

            # Convert Gmail date to readable format
            try:
                timestamp = datetime.strptime(date_str[:-6], "%a, %d %b %Y %H:%M:%S")
                timings = timestamp.strftime("%d %b %Y, %I:%M %p")
            except Exception:
                timings = "Unknown"

            subject, body, text = get_message_content(service, msg_id)
            prediction = classifier.predict(text)

            results.append({
                "from": sender,
                "subject": subject,
                "snippet": text[:150] + "...",
                "timings": timings,
                "prob": f"{prediction['avg_prob']:.2f}",
                "is_spam": "SPAM" if prediction["is_spam"] else "HAM"
            })

        return render_template("results.html", results=results)

    except Exception as e:
        flash(f"Error checking emails: {e}", "danger")
        return redirect(url_for("dashboard"))


# ------------------- REFRESH EMAILS -------------------
@app.route("/refresh_emails")
def refresh_emails():
    flash("🔄 Emails refreshed successfully!", "info")
    return redirect(url_for("check_emails"))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

@app.route("/accuracy")
def accuracy():
    try:
        # -------- Evaluate real metrics on your prepared dataset --------
        # Expected dataset from prepare_dataset.py: data/combined_spam.csv with columns: label (ham/spam), subject, from, text
        data_path = os.path.join("data", "combined_spam.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}. Run prepare_dataset.py first.")

        df = pd.read_csv(data_path)
        if 'label' not in df.columns:
            raise ValueError("'label' column not found in dataset")
        # Map labels to 0/1: 0=HAM, 1=SPAM
        label_map = {'ham': 0, 'HAM': 0, 0: 0, 'spam': 1, 'SPAM': 1, 1: 1}
        y_true = df['label'].map(label_map).astype(int).tolist()

        # Use subject + text for evaluation if available
        if 'subject' in df.columns and 'text' in df.columns:
            texts = (df['subject'].fillna('') + ' ' + df['text'].fillna('')).astype(str).tolist()
        elif 'text' in df.columns:
            texts = df['text'].fillna('').astype(str).tolist()
        else:
            raise ValueError("Dataset must contain 'text' or ('subject' and 'text') columns")

        # Transform all texts once
        X = classifier.vectorizer.transform(texts)

        # Map internal model keys to display names
        model_name_map = {"lr": "Logistic Regression", "svm": "SVM", "nb": "Naive Bayes"}
        models = [model_name_map[k] for k in classifier.models.keys() if k in model_name_map]

        # Compute per-model probabilities and predictions using thresholds
        y_preds = {}
        prob_matrix = []  # collect probabilities for ensemble average

        for key, model in classifier.models.items():
            # Skip unknown keys
            if key not in model_name_map:
                continue
            name = model_name_map[key]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                df_vals = model.decision_function(X)
                probs = 1.0 / (1.0 + np.exp(-df_vals))
            else:
                preds_binary = model.predict(X)
                probs = preds_binary.astype(float)

            prob_matrix.append(probs)
            threshold = classifier.thresholds.get(key, 0.5)
            preds = (probs >= threshold).astype(int)
            y_preds[name] = preds.tolist()

        # Ensemble: majority vote OR average prob >= ensemble_threshold
        if prob_matrix:
            probs_stack = np.vstack(prob_matrix)  # shape: (n_models, n_samples)
            avg_probs = probs_stack.mean(axis=0)
            # majority vote uses each model's thresholded predictions
            votes_stack = np.vstack([np.array(y_preds[model_name_map[k]]) for k in classifier.models.keys() if k in model_name_map])
            majority = (votes_stack.sum(axis=0) >= 2).astype(int)
            ensemble_preds = np.where(
                (avg_probs >= classifier.ensemble_threshold) | (majority == 1),
                1,
                0,
            )
            y_preds["Ensemble"] = ensemble_preds.tolist()
            models.append("Ensemble")

        # Compute metrics
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for model in models:
            preds = y_preds[model]
            accuracies.append(100 * sum([y_true[i] == preds[i] for i in range(len(y_true))]) / len(y_true))
            precisions.append(100 * precision_score(y_true, preds, zero_division=0))
            recalls.append(100 * recall_score(y_true, preds, zero_division=0))
            f1_scores.append(100 * f1_score(y_true, preds, zero_division=0))
        

        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        import matplotlib.pyplot as plt

        import matplotlib.pyplot as plt
        import io, base64

        # -------- Accuracy Bar Chart --------
        plt.rcParams.update({
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
        })
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=180, constrained_layout=False)
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#1f1f1f')
        bars = ax.bar(models, accuracies, color="#ff5733")
        ax.set_ylabel("Accuracy (%)", color='white', labelpad=6)
        ax.set_title("Model Accuracies", color='white', pad=10)
        ax.tick_params(axis='x', colors='white')
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(15)
            lbl.set_ha('right')
        ax.tick_params(axis='y', colors='white')
        ax.set_ylim(0, 100)
        # Annotate bars with values
        try:
            ax.bar_label(bars, fmt='%.1f', padding=4, color='white', fontsize=10)
        except Exception:
            pass
        plt.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.2)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
        buf.seek(0)
        accuracy_image = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # -------- Confusion Matrices --------
        cm_images = {}
        for model in models:
            cm = confusion_matrix(y_true, y_preds[model])
            fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=200, constrained_layout=False)
            fig.patch.set_facecolor('#121212')
            ax.set_facecolor('#1f1f1f')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["HAM","SPAM"])
            disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False, values_format='d')
            # Title and labels
            ax.set_title(f"{model} Confusion Matrix", color='white', fontsize=13, pad=10)
            ax.set_xlabel("Predicted label", color='white', labelpad=8)
            ax.set_ylabel("True label", color='white', labelpad=8)
            ax.tick_params(axis='both', colors='white', labelsize=11)
            # Make cell text readable (white, bold, larger)
            for text in ax.texts:
                text.set_color('white')
                text.set_fontsize(12)
                text.set_fontweight('bold')
                text.set_bbox(dict(facecolor='black', edgecolor='none', alpha=0.35, boxstyle='round,pad=0.15'))
            # Remove grid and tighten
            ax.grid(False)
            plt.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.12)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
            buf.seek(0)
            cm_images[model] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

        # -------- Precision, Recall, F1 Bar Chart --------
        fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=180, constrained_layout=False)
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#1f1f1f')
        width = 0.24
        x = list(range(len(models)))
        b1 = ax.bar([i - width for i in x], precisions, width=width, color="#33ff57", label="Precision")
        b2 = ax.bar(x, recalls, width=width, color="#33c3ff", label="Recall")
        b3 = ax.bar([i + width for i in x], f1_scores, width=width, color="#ff33aa", label="F1-Score")
        ax.set_xticks(x)
        ax.set_xticklabels(models, color='white', rotation=15, ha='right')
        ax.set_ylabel("Percentage (%)", color='white', labelpad=6)
        ax.set_title("Precision, Recall, F1-Score", color='white', pad=10)
        ax.tick_params(axis='y', colors='white')
        ax.set_ylim(0, 100)
        # Annotate values
        for bars in (b1, b2, b3):
            try:
                ax.bar_label(bars, fmt='%.1f', padding=4, color='white', fontsize=9)
            except Exception:
                pass
        leg = ax.legend(facecolor='#1f1f1f', edgecolor='white', loc='upper right', framealpha=0.9)
        for text in leg.get_texts():
            text.set_color('white')
        plt.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.25)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), bbox_inches='tight')
        buf.seek(0)
        metrics_image = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return render_template(
            "accuracy_page.html",
            accuracy_image=accuracy_image,
            cm_images=cm_images,
            metrics_image=metrics_image
        )

    except Exception as e:
        flash(f"Failed to generate accuracy page: {e}", "danger")
        return redirect(url_for("dashboard"))

# ------------------- LINK ANOTHER ACCOUNT -------------------
@app.route("/link_another")
def link_another():
    if os.path.exists("token.pickle"):
        os.remove("token.pickle")
    session.clear()
    flash("You can now link another Gmail account.", "info")
    return redirect(url_for("home"))

# ------------------- MAIN -------------------
if __name__ == "__main__":
    app.run(debug=True)
