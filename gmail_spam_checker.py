# gmail_spam_checker.py
"""
Gmail Spam Checker
Integrates SpamClassifier to read unread Gmail messages, predict spam,
and print/log results. DRY_RUN mode prevents actual Gmail modification.
"""

import base64
import os.path
import re
from email import message_from_bytes
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import time
from spam_classifier import SpamClassifier

# ---------------------- CONFIGURATION ----------------------
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
DRY_RUN = True   # set to False later when confident
CHECK_INTERVAL = 60  # seconds between checks
LOG_FILE = "processed_log.csv"

# ---------------------- GMAIL AUTH SETUP ----------------------
def authenticate_gmail(force_new_login=True):
    """Authenticate Gmail account, forcing new login if desired."""
    creds = None

    # Always force a fresh Gmail login (so user can choose account)
    if force_new_login and os.path.exists("token.pickle"):
        os.remove("token.pickle")

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If no creds or invalid, initiate OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES
            )
            # Launch OAuth in browser to choose Gmail account
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)
    return service

# ---------------------- GMAIL HELPERS ----------------------
def get_unread_messages(service):
    """Fetch unread emails (max 20 at once)"""
    results = service.users().messages().list(userId='me', labelIds=['INBOX', 'UNREAD'], maxResults=20).execute()
    messages = results.get('messages', [])
    return messages

def get_message_content(service, msg_id):
    """Extract subject and plain text body"""
    msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    payload = msg['payload']
    headers = payload.get("headers", [])
    subject = ""
    for h in headers:
        if h['name'].lower() == 'subject':
            subject = h['value']
            break

    # Decode body text
    body = ""
    if 'data' in payload.get('body', {}):
        body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
    else:
        for part in payload.get('parts', []):
            if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
    
    text = f"{subject} {body}"
    text = re.sub(r'\s+', ' ', text).strip()
    return subject, body, text

def mark_as_processed(service, msg_id, is_spam):
    """Mark messages as read or spam depending on classification"""
    if DRY_RUN:
        print(f"DRY_RUN mode: skipping Gmail label update for message {msg_id}")
        return
    if is_spam:
        service.users().messages().modify(
            userId='me', id=msg_id,
            body={'addLabelIds': ['SPAM'], 'removeLabelIds': ['INBOX']}
        ).execute()
    else:
        service.users().messages().modify(
            userId='me', id=msg_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()

# ---------------------- MAIN LOOP ----------------------
def main():
    service = authenticate_gmail()
    clf = SpamClassifier()
    print("✅ Gmail Spam Checker started.\nDRY_RUN =", DRY_RUN)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("timestamp,sender,subject,avg_prob,is_spam,reason\n")

    while True:
        messages = get_unread_messages(service)
        if not messages:
            print("No new unread emails found.")
        else:
            print(f"📬 Found {len(messages)} unread message(s).")
            for msg in messages:
                msg_id = msg['id']
                msg_detail = service.users().messages().get(userId='me', id=msg_id, format='metadata', metadataHeaders=['From']).execute()
                sender = next((h['value'] for h in msg_detail['payload']['headers'] if h['name'] == 'From'), '')
                subject, body, text = get_message_content(service, msg_id)

                result = clf.predict(text)
                decision = "SPAM" if result['is_spam'] else "HAM"
                print(f"\nFrom: {sender}\nSubject: {subject}")
                print(f"Model probabilities: {result['model_probs']}")
                print(f"Avg prob: {result['avg_prob']:.3f}  -> {decision} ({result['decision_reason']})")

                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"{time.time()},{sender},{subject},{result['avg_prob']:.4f},{decision},{result['decision_reason']}\n")

                mark_as_processed(service, msg_id, result['is_spam'])

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
