#!/usr/bin/env python3
"""
Phishing Email Detection Model (Scikit-learn)
---------------------------------------------
- Trains on a CSV of phishing / legitimate emails (or built-in demo data)
- Extracts text features (TF-IDF) + URL / keyword features
- Classifies an email as "Phishing" or "Safe"
- Prints accuracy, classification report and confusion matrix (also saved as PNG)

Usage:
    python phishing_detector.py                         # train on demo data
    python phishing_detector.py --data Phishing_Email.csv   # train on a real dataset
    python phishing_detector.py --predict "Your account is locked, verify at http://bit.ly/x1"
    python phishing_detector.py --interactive           # type emails to classify

CSV format: one column with the email text (text / Email Text / body ...) and
one label column (label / Email Type / class ...) containing phishing/safe.
"""
import argparse
import os
import random
import re
import sys
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_FILE = "phishing_model.joblib"
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly")
SUSPICIOUS_WORDS = [
    "urgent", "verify", "suspended", "locked", "immediately", "password",
    "confirm", "login", "click", "limited time", "winner", "prize", "bank",
    "account", "security alert", "update your", "expire", "act now",
    "credit card", "ssn", "refund", "unusual activity",
]
NUM_COLS = ["n_urls", "n_http", "has_ip_url", "has_shortener", "max_url_len",
            "max_host_hyphens", "at_in_url", "n_suspicious", "n_exclaim",
            "upper_ratio", "text_len"]


# ------------------------------------------------------------ features ----
def extract_features(text):
    text = str(text)
    urls = URL_RE.findall(text)
    hosts = []
    for u in urls:
        u2 = u if u.lower().startswith("http") else "http://" + u
        hosts.append((urlparse(u2).hostname or "").lower())
    low = text.lower()
    letters = [c for c in text if c.isalpha()]
    return {
        "n_urls": len(urls),
        "n_http": sum(u.lower().startswith("http://") for u in urls),
        "has_ip_url": int(any(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", h) for h in hosts)),
        "has_shortener": int(any(h in SHORTENERS for h in hosts)),
        "max_url_len": max((len(u) for u in urls), default=0),
        "max_host_hyphens": max((h.count("-") for h in hosts), default=0),
        "at_in_url": int(any("@" in u for u in urls)),
        "n_suspicious": sum(w in low for w in SUSPICIOUS_WORDS),
        "n_exclaim": text.count("!"),
        "upper_ratio": sum(c.isupper() for c in letters) / len(letters) if letters else 0,
        "text_len": len(text),
    }


def build_frame(texts):
    df = pd.DataFrame({"text": list(texts)})
    df["clean"] = df["text"].map(lambda t: URL_RE.sub(" urltoken ", str(t)))
    feats = pd.DataFrame([extract_features(t) for t in df["text"]])
    return pd.concat([df, feats], axis=1)


def red_flags(text):
    f = extract_features(text)
    flags = []
    if f["has_ip_url"]:
        flags.append("link uses a raw IP address")
    if f["has_shortener"]:
        flags.append("shortened link (hides real destination)")
    if f["n_http"]:
        flags.append(f"{f['n_http']} link(s) without HTTPS")
    if f["max_host_hyphens"] >= 2:
        flags.append("link domain has many hyphens")
    if f["at_in_url"]:
        flags.append("'@' inside a link")
    if f["n_suspicious"] >= 3:
        flags.append(f"{f['n_suspicious']} suspicious keywords")
    if f["n_exclaim"] >= 3:
        flags.append("excessive exclamation marks")
    return flags


# ---------------------------------------------------------------- data ----
def make_demo_data(n=1200, seed=42):
    """Synthetic emails with overlapping wording so the task is not trivial.
    Replace with a real dataset (e.g. Kaggle 'Phishing Email Dataset')."""
    rnd = random.Random(seed)
    brands = ["PayPal", "Amazon", "Netflix", "Microsoft", "Apple", "your bank", "DHL"]
    greet = ["Dear customer,", "Hello,", "Hi,", "Dear user,", "Good morning,", "Dear {n},"]
    names = ["Priya", "Arun", "Kavya", "Ravi", "Meena", "Sam"]
    phish_core = [
        "We detected unusual activity on your {b} account.",
        "Your {b} account has been suspended due to a security issue.",
        "Please confirm your password to avoid losing access.",
        "You have won a prize! Claim your reward now.",
        "Your payment of $%d could not be processed. Update your billing details.",
        "Your invoice is attached, please review your account.",
        "A refund is waiting for you. Confirm your bank details.",
        "Your mailbox is almost full. Login to keep your emails.",
    ]
    phish_urgent = ["Act now or your account will be locked within 24 hours!",
                    "Verify immediately to avoid suspension!", "This is urgent.",
                    "Limited time offer, click the link below."]
    legit_core = [
        "Please find the meeting notes from yesterday's project review.",
        "Your {b} order has shipped and will arrive on Friday.",
        "Reminder: the team lunch is on Thursday at 1 pm.",
        "Here is the updated report for this quarter, let me know your feedback.",
        "Your {b} receipt for the subscription is attached.",
        "The class schedule for next week has been updated.",
        "Thanks for your help with the presentation, it went well.",
        "Your password was changed successfully. If this was you, no action is needed.",
        "Security notice: scheduled server maintenance tonight at 11 pm.",
        "Please review the attached invoice for the project and approve it by Monday.",
    ]
    legit_urgent = ["It is fairly urgent, so please reply today.", "Please confirm if you can attend."]
    good_domains = ["https://www.github.com/org/repo", "https://docs.google.com/document/d/abc",
                    "https://www.amazon.in/orders", "https://meet.google.com/xyz-abcd-efg",
                    "https://www.linkedin.com/in/someone", "https://www.microsoft.com/en-in/"]
    close = ["Regards,", "Thanks,", "Best,", "Customer Support Team", "Security Team", "Admin"]

    def bad_link():
        b = rnd.choice(["paypal", "amazon", "netflix", "microsoft", "apple", "bank"])
        return rnd.choice([
            f"http://{b}-secure-verify.com/login?id={rnd.randint(1000, 99999)}",
            f"http://{rnd.randint(11,220)}.{rnd.randint(0,255)}.{rnd.randint(0,255)}.{rnd.randint(1,254)}/{b}/signin",
            f"http://bit.ly/{rnd.choice('abcdefgh')}{rnd.randint(100, 999)}",
            f"http://{b}.account-update.{rnd.choice(['xyz', 'top', 'info'])}/confirm",
            f"https://{b}-support-center.{rnd.choice(['com', 'net'])}/verify",
        ])

    rows = []
    for i in range(n):
        phishing = i % 2 == 0
        b = rnd.choice(brands)
        parts = [rnd.choice(greet).format(n=rnd.choice(names))]
        if phishing:
            core = rnd.choice(phish_core)
            parts.append(core % rnd.randint(20, 900) if "%d" in core else core.format(b=b))
            if rnd.random() < 0.55:
                parts.append(rnd.choice(phish_urgent))
            if rnd.random() < 0.85:
                parts.append("Click here: " + bad_link())
        else:
            parts.append(rnd.choice(legit_core).format(b=b))
            if rnd.random() < 0.2:
                parts.append(rnd.choice(legit_urgent))
            if rnd.random() < 0.45:
                parts.append("Link: " + rnd.choice(good_domains))
            if rnd.random() < 0.04:
                parts.append("Details: " + bad_link())  # occasional noisy legit mail
        parts.append(rnd.choice(close))
        rows.append((" ".join(parts), int(phishing)))
    return pd.DataFrame(rows, columns=["text", "label"])


def load_data(path):
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if c.lower() in
                 ("text", "email text", "body", "email", "message", "content")), None)
    lcol = next((c for c in df.columns if c.lower() in
                 ("label", "email type", "class", "target", "type")), None)
    if tcol is None or lcol is None:
        sys.exit(f"Could not find text/label columns. Found: {list(df.columns)}")
    df = df[[tcol, lcol]].dropna().rename(columns={tcol: "text", lcol: "label"})
    lab = df["label"].astype(str).str.lower()
    df["label"] = lab.str.contains(r"phish|spam|malicious|^1$").astype(int)
    return df.reset_index(drop=True)


# ------------------------------------------------------------ training ----
def make_pipeline(model):
    pre = ColumnTransformer([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000,
                                  stop_words="english"), "clean"),
        ("num", StandardScaler(), NUM_COLS),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def plot_cm(cm, name, path="confusion_matrix.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Safe", "Phishing"])
    ax.set_yticks([0, 1], ["Safe", "Phishing"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({name})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Confusion matrix image saved to {path}")


def train(df):
    print(f"Dataset: {len(df)} emails | phishing={int(df.label.sum())} "
          f"safe={int((1 - df.label).sum())}")
    X = build_frame(df["text"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    best = None
    for name, model in candidates.items():
        pipe = make_pipeline(model).fit(X_tr, y_tr)
        acc = accuracy_score(y_te, pipe.predict(X_te))
        print(f"  {name:20s} accuracy = {acc:.4f}")
        if best is None or acc > best[2]:
            best = (name, pipe, acc)

    name, pipe, acc = best
    pred = pipe.predict(X_te)
    cm = confusion_matrix(y_te, pred)
    print(f"\nBest model: {name}  (test accuracy {acc * 100:.2f}%)")
    print("\nClassification report:")
    print(classification_report(y_te, pred, target_names=["Safe", "Phishing"], digits=3))
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=["Actual Safe", "Actual Phishing"],
                       columns=["Pred Safe", "Pred Phishing"]))
    plot_cm(cm, name)
    joblib.dump(pipe, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    return pipe


def classify(pipe, text):
    X = build_frame([text])
    p = float(pipe.predict_proba(X)[0][1])
    label = "Phishing" if p >= 0.5 else "Safe"
    print(f"\nResult : {label}   (phishing probability {p * 100:.1f}%)")
    flags = red_flags(text)
    if flags:
        print("Red flags: " + "; ".join(flags))


def main():
    ap = argparse.ArgumentParser(description="Phishing Email Detector")
    ap.add_argument("--data", help="CSV file with email text and label")
    ap.add_argument("--predict", help="classify this email text")
    ap.add_argument("--interactive", action="store_true", help="classify emails you type")
    a = ap.parse_args()

    if (a.predict or a.interactive) and os.path.exists(MODEL_FILE) and not a.data:
        pipe = joblib.load(MODEL_FILE)
    else:
        if a.data:
            df = load_data(a.data)
        else:
            print("No --data given: using built-in DEMO data (synthetic). "
                  "Use a real dataset for real results.\n")
            df = make_demo_data()
        pipe = train(df)

    if a.predict:
        classify(pipe, a.predict)
    if a.interactive:
        print("\nPaste an email (single line) and press Enter. Empty line to quit.")
        while True:
            t = input("\nEmail > ").strip()
            if not t:
                break
            classify(pipe, t)


if __name__ == "__main__":
    main()
