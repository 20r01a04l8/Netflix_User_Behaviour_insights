#!/usr/bin/env python3
"""
clean_reviews.py — FINAL BASE64 VERSION (fixed)

Features / fixes applied:
 - Added missing imports (time, math, random, traceback, argparse)
 - Implemented safe_set_with_dataframe with proper imports and error handling
 - Replaced direct set_with_dataframe(ws, df) call with safe_set_with_dataframe
 - Added CLI flag --no-sheets to skip Sheets upload (useful in CI)
 - Writes CSV fallback to OUT_CSV_PATH if Sheets upload fails
 - Improved logging and defensive checks
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

# Missing imports fixed
import time
import math
import random
import traceback
import argparse
from gspread.exceptions import APIError

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
    If still failing, export CSV to OUT_CSV_PATH and return False.
    """
    # 1) quick attempt with set_with_dataframe + retries
    for attempt in range(1, max_retries + 1):
        try:
            LOGGER.info("Attempt %d/%d: set_with_dataframe -> writing full frame", attempt, max_retries)
            set_with_dataframe(worksheet, df)
            LOGGER.info("Upload succeeded with set_with_dataframe.")
            return True
        except APIError as e:
            msg = str(e)
            LOGGER.warning("APIError on attempt %d: %s", attempt, msg)
            # exponential backoff + jitter
            sleep_sec = min(60, (2 ** attempt)) + random.random()
            LOGGER.info("Sleeping %.1fs before retry...", sleep_sec)
            time.sleep(sleep_sec)
        except Exception as e:
            LOGGER.warning("Unexpected error on attempt %d: %s", attempt, e)
            time.sleep(min(60, (2 ** attempt)))

    # 2) fallback: chunked upload using worksheet.append_rows (less payload per request)
    LOGGER.info("Falling back to chunked upload using append_rows (safer for big sheets).")
    try:
        # clear existing content (optional)
        try:
            worksheet.clear()
        except Exception as e:
            LOGGER.warning("Unable to clear worksheet before chunked upload: %s", e)

        headers = list(df.columns)
        # append headers
        worksheet.append_row(headers, value_input_option='USER_ENTERED')
        n_rows = len(df)
        n_chunks = max(1, math.ceil(n_rows / chunk_size_rows))
        for i in range(n_chunks):
            start = i * chunk_size_rows
            end = min(start + chunk_size_rows, n_rows)
            chunk_df = df.iloc[start:end]
            values = chunk_df.values.tolist()
            LOGGER.info("Uploading chunk %d/%d rows %d:%d", i + 1, n_chunks, start, end)
            worksheet.append_rows(values, value_input_option='USER_ENTERED')
        LOGGER.info("Chunked upload completed successfully.")
        return True
    except APIError as e:
        LOGGER.error("Chunked upload APIError: %s", e)
    except Exception as e:
        LOGGER.error("Chunked upload failed with exception: %s", e)
        traceback.print_exc()

    # 3) final fallback: write CSV to OUT_CSV_PATH and return False
    fallback_path = OUT_CSV_PATH
    try:
        LOGGER.info("Saving CSV fallback to %s", fallback_path)
        df.to_csv(fallback_path, index=False)
        LOGGER.info("Saved fallback CSV.")
    except Exception as e:
        LOGGER.error("Failed to save fallback CSV: %s", e)

    return False


def clean_text_basic(s):
    if not s:
        return ""
    s = str(s)
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^A-Za-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


def pipeline(no_sheets: bool = False):
    """
    Main pipeline:
      - download dataset from Kaggle (requires KAGGLE_SLUG + kaggle.json)
      - basic cleaning: text, lemmatization via spaCy, VADER sentiment
      - save CSV to OUT_CSV_PATH
      - optionally upload to Google Sheets (GCP_SA_FILE + SPREADSHEET_ID required)
    """

    if not KAGGLE_SLUG:
        LOGGER.error("KAGGLE_DATASET_SLUG is not set (env var KAGGLE_DATASET_SLUG). Exiting.")
        raise RuntimeError("KAGGLE_DATASET_SLUG is required to download dataset from Kaggle.")

    LOGGER.info("DOWNLOADING DATASET...")
    csv_path = download_kaggle()

    LOGGER.info("Loading CSV...")
    df = pd.read_csv(csv_path)

    # infer columns
    text_cols = [c for c in df.columns if "review" in c.lower() or "content" in c.lower()]
    date_cols = [c for c in df.columns if "date" in c.lower() or c.lower() == "at"]
    rating_cols = [c for c in df.columns if "rating" in c.lower() or "score" in c.lower()]

    if not text_cols:
        LOGGER.warning("No text-like column found. Using first column as text.")
        TEXT = df.columns[0]
    else:
        TEXT = text_cols[0]

    DATE = date_cols[0] if date_cols else None
    RATING = rating_cols[0] if rating_cols else None

    LOGGER.info("Text column: %s, Date column: %s, Rating column: %s", TEXT, DATE, RATING)

    # cleaning
    df["clean_text"] = df[TEXT].fillna("").apply(clean_text_basic)

    # load spaCy
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception as e:
            LOGGER.warning("spaCy model en_core_web_sm not available; attempting to download. Error: %s", e)
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Lemmatize & remove stop words
    def lem_text(t):
        try:
            doc = nlp(t)
            return " ".join([token.lemma_ for token in doc if not token.is_stop and token.lemma_.strip()])
        except Exception:
            return t

    df["lemma"] = df["clean_text"].apply(lambda t: lem_text(str(t)))

    # sentiment (VADER)
    df["sentiment"] = df[TEXT].fillna("").apply(lambda x: sentiment.polarity_scores(str(x))["compound"])

    # rating normalization (if rating column available)
    if RATING:
        try:
            df[RATING] = pd.to_numeric(df[RATING], errors="coerce")
            scaler = MinMaxScaler()
            df["norm_rating"] = scaler.fit_transform(df[[RATING]].fillna(0))
        except Exception as e:
            LOGGER.warning("Could not normalize rating column: %s", e)

    # outlier detection: very long or very short reviews (length-based)
    df["review_len"] = df["clean_text"].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
    q_low, q_high = df["review_len"].quantile([0.01, 0.99])
    df["len_outlier"] = ((df["review_len"] < q_low) | (df["review_len"] > q_high))

    # Save cleaned CSV
    try:
        df.to_csv(OUT_CSV_PATH, index=False)
        LOGGER.info("Saved cleaned CSV to %s", OUT_CSV_PATH)
    except Exception as e:
        LOGGER.error("Failed to save cleaned CSV to %s: %s", OUT_CSV_PATH, e)
        raise

    # If user requested no_sheets, skip Sheets upload
    if no_sheets:
        LOGGER.info("--no-sheets flag set; skipping Google Sheets upload.")
        return

    # Google Sheets upload
    if not GCP_SA_FILE:
        LOGGER.warning("GCP SA JSON (GCP_SA_FILE) missing. Skipping Sheets upload.")
        return

    if not SPREADSHEET_ID:
        LOGGER.warning("SPREADSHEET_ID missing. Skipping Sheets upload.")
        return

    try:
        client = gspread.service_account(filename=GCP_SA_FILE)
    except Exception as e:
        LOGGER.error("Failed to create gspread client from service account file %s: %s", GCP_SA_FILE, e)
        LOGGER.info("Skipping Sheets upload and keeping CSV on disk.")
        return

    try:
        sh = client.open_by_key(SPREADSHEET_ID)
    except Exception as e:
        LOGGER.error("Failed to open spreadsheet id %s: %s", SPREADSHEET_ID, e)
        LOGGER.info("Skipping Sheets upload and keeping CSV on disk.")
        return

    # Get or create worksheet
    try:
        try:
            ws = sh.worksheet(SHEET_NAME)
            LOGGER.info("Found existing worksheet '%s'", SHEET_NAME)
            # optionally clear existing to avoid stray cells (we will write full df)
            try:
                ws.clear()
            except Exception as e:
                LOGGER.warning("Could not clear worksheet: %s", e)
        except gspread.WorksheetNotFound:
            LOGGER.info("Worksheet '%s' not found. Creating new.", SHEET_NAME)
            ws = sh.add_worksheet(title=SHEET_NAME, rows="2000", cols=str(max(20, len(df.columns))))
    except Exception as e:
        LOGGER.error("Failed to get or create worksheet: %s", e)
        LOGGER.info("Skipping Sheets upload and keeping CSV on disk.")
        return

    # Attempt safe upload with retries and chunk fallback
    success = safe_set_with_dataframe(ws, df)
    if not success:
        LOGGER.warning("Upload to Google Sheets failed after retries. CSV remained at %s", OUT_CSV_PATH)
        # Do not hard-fail the whole job; CSV is saved and can be uploaded as an artifact.
        return

    LOGGER.info("Google Sheet updated successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean reviews and optionally upload to Google Sheets.")
    parser.add_argument("--no-sheets", action="store_true", help="Skip uploading to Google Sheets (useful for CI).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        pipeline(no_sheets=args.no_sheets)
    except Exception as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        traceback.print_exc()
        # Save any last CSV if present to OUT_CSV_PATH (defensive)
        try:
            # if df exists in locals, write it
            if "df" in locals() and isinstance(df, pd.DataFrame):
                df.to_csv(OUT_CSV_PATH, index=False)
                LOGGER.info("Saved intermediate CSV to %s after failure.", OUT_CSV_PATH)
        except Exception as e:
            LOGGER.error("Failed to save intermediate CSV after exception: %s", e)
        # propagate non-zero exit so CI can detect the failure
        sys.exit(1)
