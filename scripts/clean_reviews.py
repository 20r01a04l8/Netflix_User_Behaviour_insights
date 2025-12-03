#!/usr/bin/env python3
"""
clean_reviews.py — FINAL BASE64 VERSION
"""

import os
import sys
import json
import logging
import subprocess
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import gspread
from gspread_dataframe import set_with_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

HOME = os.getenv("HOME", "/home/runner")
WORKSPACE = os.getenv("GITHUB_WORKSPACE", os.getcwd())

KAGGLE_SLUG = os.getenv("KAGGLE_DATASET_SLUG", "")
KAGGLE_FILE = os.getenv("KAGGLE_FILE_NAME", "")
KAGGLE_CONFIG_DIR = os.getenv("KAGGLE_CONFIG_DIR", f"{HOME}/.kaggle")

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
SHEET_NAME = os.getenv("SHEET_NAME", "Sheet1")
GCP_SA_FILE = os.getenv("GCP_SA_FILE")

OUT_CSV_PATH = os.getenv("OUT_CSV_PATH", f"{WORKSPACE}/outputs/cleaned_reviews.csv")

os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)
os.makedirs(f"{WORKSPACE}/data", exist_ok=True)

nlp = None
sentiment = SentimentIntensityAnalyzer()


def validate_kaggle():
    """Check if kaggle.json exists and has correct fields."""
    cfg = f"{KAGGLE_CONFIG_DIR}/kaggle.json"
    if not os.path.exists(cfg):
        raise RuntimeError(f"Missing kaggle.json at {cfg}")

    with open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "username" not in data or "key" not in data:
        raise RuntimeError("kaggle.json missing username/key")

    LOGGER.info("Kaggle config OK (username masked: %s***, key length: %d)",
                data["username"][:2], len(data["key"]))


def download_kaggle():
    validate_kaggle()

    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_SLUG,
        "-p", f"{WORKSPACE}/data",
        "--unzip",
        "--force"
    ]

    if KAGGLE_FILE:
        cmd += ["-f", KAGGLE_FILE]

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = KAGGLE_CONFIG_DIR

    LOGGER.info("Running Kaggle download...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    LOGGER.info("Kaggle stdout: %s", result.stdout[:300])
    LOGGER.info("Kaggle stderr: %s", result.stderr[:300])

    if result.returncode != 0:
        raise RuntimeError("Kaggle download failed")

    # find csv
    for f in os.listdir(f"{WORKSPACE}/data"):
        if f.lower().endswith(".csv"):
            return f"{WORKSPACE}/data/{f}"

    raise RuntimeError("No CSV found after download")

def safe_set_with_dataframe(worksheet, df, max_retries=5, chunk_size_rows=500):
    """
    Try to write df to worksheet with retries.
    If a 500/5xx persists, fallback to chunked uploads (append_rows) which are less likely to trigger
    the Sheets 500 for large single requests.
    If still failing, export CSV to outputs/cleaned_reviews.csv and return False.
    """
    # 1) quick attempt with set_with_dataframe + retries
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt}/{max_retries}: set_with_dataframe -> writing full frame")
            set_with_dataframe(worksheet, df)
            print("Upload succeeded with set_with_dataframe.")
            return True
        except APIError as e:
            # check status code if available (gspread wraps HTTP errors)
            msg = str(e)
            print(f"APIError on attempt {attempt}: {msg}")
            # exponential backoff
            sleep = min(60, 2 ** attempt)
            print(f"Sleeping {sleep}s before retry...")
            time.sleep(sleep)
        except Exception as e:
            print(f"Unexpected error on attempt {attempt}: {e}")
            time.sleep(min(60, 2 ** attempt))

    # 2) fallback: chunked upload using worksheet.append_rows (less payload per request)
    print("Falling back to chunked upload using append_rows (safer for big sheets).")
    try:
        # clear existing content (optional)
        try:
            worksheet.clear()
        except Exception as e:
            print("Warning: unable to clear worksheet before chunked upload:", e)

        headers = list(df.columns)
        # append headers
        worksheet.append_row(headers, value_input_option='USER_ENTERED')
        n_rows = len(df)
        n_chunks = math.ceil(n_rows / chunk_size_rows)
        for i in range(n_chunks):
            start = i * chunk_size_rows
            end = min(start + chunk_size_rows, n_rows)
            chunk_df = df.iloc[start:end]
            values = chunk_df.values.tolist()
            # prepend no header here, headers already written
            print(f"Uploading chunk {i+1}/{n_chunks} rows {start}:{end}")
            # append_rows can accept list of rows
            worksheet.append_rows(values, value_input_option='USER_ENTERED')
        print("Chunked upload completed successfully.")
        return True
    except APIError as e:
        print("Chunked upload APIError:", e)
    except Exception as e:
        print("Chunked upload failed with exception:", e)

    # 3) final fallback: write CSV to outputs/ and return False
    fallback_path = 'outputs/cleaned_reviews.csv'
    try:
        print(f"Saving CSV fallback to {fallback_path}")
        df.to_csv(fallback_path, index=False)
        print("Saved fallback CSV.")
    except Exception as e:
        print("Failed to save fallback CSV:", e)

    return False

# Usage: replace the previous set_with_dataframe(ws, df) call with:
success = safe_set_with_dataframe(ws, df)
if not success:
    print("Upload to Google Sheets failed after retries. CSV saved to outputs/cleaned_reviews.csv")
    # exit gracefully so CI can still collect artifacts and not crash the workflow hard
    # choose exit code 0 to avoid failing the whole job if you prefer artifacts to be saved.
    # If you want the job to fail (so a human checks), use non-zero exit code.
    # Here we exit 0 so the workflow continues to upload artifacts and logs.
    sys.exit(0)

def clean_text_basic(s):
    if not s:
        return ""
    s = str(s)
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^A-Za-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


def pipeline():
    if not SPREADSHEET_ID:
        raise RuntimeError("SPREADSHEET_ID is missing")

    LOGGER.info("DOWNLOADING DATASET...")
    csv_path = download_kaggle()

    LOGGER.info("Loading CSV...")
    df = pd.read_csv(csv_path)

    text_cols = [c for c in df.columns if "review" in c.lower() or "content" in c.lower()]
    date_cols = [c for c in df.columns if "date" in c.lower() or c.lower() == "at"]
    rating_cols = [c for c in df.columns if "rating" in c.lower() or "score" in c.lower()]

    TEXT = text_cols[0]
    DATE = date_cols[0] if date_cols else None
    RATING = rating_cols[0] if rating_cols else None

    df["clean_text"] = df[TEXT].fillna("").apply(clean_text_basic)

    # load spaCy
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    df["lemma"] = df["clean_text"].apply(lambda t: " ".join([token.lemma_ for token in nlp(t) if not token.is_stop]))

    df["sentiment"] = df[TEXT].fillna("").apply(lambda x: sentiment.polarity_scores(str(x))["compound"])

    df.to_csv(OUT_CSV_PATH, index=False)
    LOGGER.info("Saved cleaned CSV to %s", OUT_CSV_PATH)

    # Google Sheets upload
    if not os.path.exists(GCP_SA_FILE):
        LOGGER.warning("GCP SA JSON missing. Skipping Sheets upload.")
        return

    client = gspread.service_account(filename=GCP_SA_FILE)
    sh = client.open_by_key(SPREADSHEET_ID)

    try:
        ws = sh.worksheet(SHEET_NAME)
        ws.clear()
    except:
        ws = sh.add_worksheet(title=SHEET_NAME, rows="2000", cols="20")

    set_with_dataframe(ws, df)
    LOGGER.info("Google Sheet updated successfully.")


if __name__ == "__main__":
    pipeline()

