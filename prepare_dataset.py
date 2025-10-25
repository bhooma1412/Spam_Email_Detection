# prepare_dataset.py
import os
import email
import pandas as pd
from tqdm import tqdm

# Folders you already have
DATA_DIRS = {
    'ham': ['easy_ham', 'hard_ham'],
    'spam': ['spam_2']
}

records = []

for label, dirs in DATA_DIRS.items():
    for d in dirs:
        folder_path = os.path.join(d)
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        print(f"📂 Processing folder: {folder_path}")
        for fname in tqdm(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, fname)
            if not os.path.isfile(file_path):
                continue
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    msg = email.message_from_file(f)
                    subject = msg.get('Subject', '')
                    sender = msg.get('From', '')
                    # Extract body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == 'text/plain':
                                body += part.get_payload(decode=True).decode('latin-1', errors='ignore')
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode('latin-1', errors='ignore')
                        except:
                            body = msg.get_payload()

                    records.append({
                        'label': label,
                        'subject': subject,
                        'from': sender,
                        'text': body
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Create data folder if not exist
os.makedirs('data', exist_ok=True)

df = pd.DataFrame(records)
df.to_csv('data/combined_spam.csv', index=False, quoting=1, escapechar='\\', encoding='utf-8')
print(f"\n✅ Dataset saved to data/combined_spam.csv with {len(df)} emails.")
print(df['label'].value_counts())
