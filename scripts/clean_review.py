#!/usr/bin/env python3
"""
clean_reviews.py — FINAL BASE64 VERSION (fixed + Sheets-safe)

Features:
 - Robust Kaggle download (uses KAGGLE_CONFIG_DIR)
 - Text cleaning + spaCy lemmatization
 - VADER sentiment
 - Rating normalization and outlier detection
 - Sanitizes DataFrame for Google Sheets (no NaN/inf; converts numpy types to native)
 - set_with_dataframe with retries and chunked append_rows fallback
 - --no-sheets CLI flag to skip Google Sheets upload (useful for CI)
 - CSV fallback to OUT_CSV_PATH if Sheets upload fails
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

# Additional imports
import time
import math
import random
import traceback
import argparse
from gspread.exceptions import APIError
from datetime import datetime, date, time as dtime

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

# Ensure dirs exist
os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)
os.makedirs(f"{WORKSPACE}/data", exist_ok=True)

nlp = None
sentiment = SentimentIntensityAnalyzer()


# ----------------------------
# Kaggle helpers
# ----------------------------
def validate_kaggle():
    """Check if kaggle.json exists and has correct fields."""
    cfg = f"{KAGGLE_CONFIG_DIR}/kaggle.json"
    if not os.path.exists(cfg):
        raise RuntimeError(f"Missing kaggle.json at {cfg}")

    with open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "username" not in data or "key" not in data:
        raise RuntimeError("kaggle.json missing username/key")

    LOGGER.info(
        "Kaggle config OK (username masked: %s***, key length: %d)",
        str(data.get("username", ""))[:2],
        len(str(data.get("key", ""))),
    )


def download_kaggle():
    """Download dataset using kaggle CLI; return path to first CSV found."""
    validate_kaggle()

    target_dir = os.path.join(WORKSPACE, "data")
    os.makedirs(target_dir, exist_ok=True)

    # Build command
    base_cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", target_dir, "--unzip", "--force"]
    if KAGGLE_FILE:
        base_cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-f", KAGGLE_FILE, "-p", target_dir, "--unzip", "--force"]

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = KAGGLE_CONFIG_DIR

    LOGGER.info("Running Kaggle download...")
    proc = subprocess.run(base_cmd, env=env, capture_output=True, text=True)
    LOGGER.info("kaggle stdout (trunc): %s", (proc.stdout or "")[:1000])
    LOGGER.info("kaggle stderr (trunc): %s", (proc.stderr or "")[:1000])

    if proc.returncode == 0:
        # find CSV
        for f in os.listdir(target_dir):
            if f.lower().endswith(".csv"):
                return os.path.join(target_dir, f)
        # if no CSV found, continue to list files
    else:
        LOGGER.warning("Initial download returned non-zero; will try listing files if available.")

    # If we reach here, attempt to list files and pick a CSV candidate
    list_cmd = ["kaggle", "datasets", "files", "-d", KAGGLE_SLUG]
    proc_list = subprocess.run(list_cmd, env=env, capture_output=True, text=True)
    LOGGER.info("kaggle files stdout (trunc): %s", (proc_list.stdout or "")[:2000])
    LOGGER.info("kaggle files stderr (trunc): %s", (proc_list.stderr or "")[:1000])

    candidates = []
    out = proc_list.stdout or ""
    for m in re.finditer(r"([^\s,']+\.csv)", out, flags=re.IGNORECASE):
        fname = m.group(1).strip().strip("()[],'\"")
        candidates.append(fname)

    # try JSON output if still empty
    if not candidates:
        proc_json = subprocess.run(["kaggle", "datasets", "files", "-d", KAGGLE_SLUG, "--json"], env=env, capture_output=True, text=True)
        if proc_json.returncode == 0 and proc_json.stdout:
            try:
                j = json.loads(proc_json.stdout)
                for obj in j:
                    name = obj.get("name") or obj.get("ref") or ""
                    if name.lower().endswith(".csv"):
                        candidates.append(name)
            except Exception:
                LOGGER.debug("Could not parse JSON from kaggle files --json output.")

    if not candidates:
        raise RuntimeError("No CSV file candidates found in dataset. Check KAGGLE_DATASET_SLUG and dataset contents. Raw list output:\n" + (proc_list.stdout or proc_list.stderr or ""))

    # select first candidate and download
    candidate = candidates[0]
    LOGGER.info("Selected file to download: %s", candidate)
    cmd2 = ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-f", candidate, "-p", target_dir, "--unzip", "--force"]
    proc2 = subprocess.run(cmd2, env=env, capture_output=True, text=True)
    LOGGER.info("download stdout (trunc): %s", (proc2.stdout or "")[:1000])
    LOGGER.info("download stderr (trunc): %s", (proc2.stderr or "")[:1000])
    if proc2.returncode != 0:
        raise RuntimeError(f"Failed to download candidate file '{candidate}'. See logs above.")
    # return first CSV
    for f in os.listdir(target_dir):
        if f.lower().endswith(".csv"):
            return os.path.join(target_dir, f)
    # fallback
    files = [os.path.join(target_dir, f) for f in os.listdir(target_dir)]
    if files:
        return files[0]
    raise RuntimeError("Download succeeded but no files found in data directory.")


# ----------------------------
# Sanitization helpers
# ----------------------------
def is_datetime_like(val):
    return isinstance(val, (pd.Timestamp, datetime, date, dtime))


def to_native_value(v):
    """
    Convert pandas/numpy scalars and datetimes to JSON-friendly native Python types.
    - NaN/inf/-inf -> "" (empty string)
    - numpy ints/floats -> native int/float
    - datetimes -> ISO string
    """
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass

    if is_datetime_like(v):
        try:
            return v.isoformat()
        except Exception:
            return str(v)

    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        fv = float(v)
        if not np.isfinite(fv):
            return ""
        return fv
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if hasattr(v, "item"):
        try:
            val = v.item()
            return to_native_value(val)
        except Exception:
            pass
    try:
        return str(v)
    except Exception:
        return ""


def sanitize_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df that is safe to be JSON-serialized for Google Sheets.
    - replace inf/-inf with NaN
    - convert datetimes to ISO strings
    - fill numeric NaN with 0
    - fill object NaN with ""
    - convert numpy scalars to Python native types
    """
    df2 = df.copy()
    df2.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Convert datetime-like cols to ISO strings
    for col in df2.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df2[col]) or df2[col].apply(lambda x: isinstance(x, (pd.Timestamp, datetime, date, dtime))).any():
                df2[col] = df2[col].apply(lambda x: x.isoformat() if is_datetime_like(x) else x)
        except Exception:
            pass

    # Numeric columns: replace NaN with 0 (statistical fields) or choose other sentinel
    for col in df2.columns:
        try:
            if pd.api.types.is_numeric_dtype(df2[col]):
                df2[col] = pd.to_numeric(df2[col], errors="coerce")
                df2[col] = df2[col].apply(lambda x: 0 if not np.isfinite(x) else x)
            else:
                df2[col] = df2[col].where(df2[col].notna(), "")
        except Exception:
            # If there's an error converting, fallback to string fill
            df2[col] = df2[col].where(df2[col].notna(), "")

    # Convert numpy scalar types to Python natives
    def _convert_cell(x):
        return to_native_value(x)

    df_safe = df2.applymap(_convert_cell)
    return df_safe


# ----------------------------
# Safe upload helper
# ----------------------------
def safe_set_with_dataframe(worksheet, df: pd.DataFrame, max_retries=5, chunk_size_rows=500) -> bool:
    """
    Upload df to worksheet safely:
      1) Try set_with_dataframe with retries.
      2) If persistent 5xx errors, fallback to chunked append_rows with sanitized values.
      3) If everything fails, save CSV to OUT_CSV_PATH and return False.
    """

    df_safe = sanitize_for_sheets(df)

    # 1) Try single call
    for attempt in range(1, max_retries + 1):
        try:
            LOGGER.info("Attempt %d/%d: set_with_dataframe -> writing full frame", attempt, max_retries)
            set_with_dataframe(worksheet, df_safe, include_index=False, include_column_header=True, resize=True)
            LOGGER.info("Upload succeeded with set_with_dataframe.")
            return True
        except APIError as e:
            msg = str(e)
            LOGGER.warning("APIError on attempt %d: %s", attempt, msg)
            sleep_sec = min(60, (2 ** attempt)) + random.random()
            LOGGER.info("Sleeping %.1fs before retry...", sleep_sec)
            time.sleep(sleep_sec)
        except Exception as e:
            LOGGER.warning("Unexpected error on attempt %d: %s", attempt, e)
            time.sleep(min(60, (2 ** attempt)))

    # 2) Fallback to chunked append_rows
    LOGGER.info("Falling back to chunked upload using append_rows (safer for big sheets).")
    try:
        try:
            worksheet.clear()
        except Exception as e:
            LOGGER.warning("Unable to clear worksheet before chunked upload: %s", e)

        headers = list(df_safe.columns)
        worksheet.append_row(headers, value_input_option="USER_ENTERED")

        n_rows = len(df_safe)
        n_chunks = max(1, math.ceil(n_rows / chunk_size_rows))
        for i in range(n_chunks):
            start = i * chunk_size_rows
            end = min(start + chunk_size_rows, n_rows)
            chunk_df = df_safe.iloc[start:end]

            values = []
            for row in chunk_df.itertuples(index=False, name=None):
                row_vals = [to_native_value(cell) for cell in row]
                values.append(row_vals)

            LOGGER.info("Uploading chunk %d/%d rows %d:%d", i + 1, n_chunks, start, end)
            worksheet.append_rows(values, value_input_option="USER_ENTERED")
        LOGGER.info("Chunked upload completed successfully.")
        return True
    except APIError as e:
        LOGGER.error("Chunked upload APIError: %s", e)
    except Exception as e:
        LOGGER.error("Chunked upload failed with exception: %s", e)
        traceback.print_exc()

    # 3) Final fallback: save CSV
    try:
        LOGGER.info("Saving CSV fallback to %s", OUT_CSV_PATH)
        df.to_csv(OUT_CSV_PATH, index=False)
        LOGGER.info("Saved fallback CSV.")
    except Exception as e:
        LOGGER.error("Failed to save fallback CSV: %s", e)

    return False


# ----------------------------
# Text cleaning helpers
# ----------------------------
def clean_text_basic(s):
    if not s:
        return ""
    s = str(s)
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"[^A-Za-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


# ----------------------------
# Main pipeline
# ----------------------------
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
    df = pd.read_csv(csv_path, low_memory=False)

    # infer columns
    text_cols = [c for c in df.columns if "review" in c.lower() or "content" in c.lower() or "comment" in c.lower()]
    date_cols = [c for c in df.columns if "date" in c.lower() or c.lower() == "at" or "time" in c.lower()]
    rating_cols = [c for c in df.columns if "rating" in c.lower() or "score" in c.lower() or "stars" in c.lower()]

    if not text_cols:
        LOGGER.warning("No text-like column found. Using first column as text.")
        TEXT = df.columns[0]
    else:
        TEXT = text_cols[0]

    DATE = date_cols[0] if date_cols else None
    RATING = rating_cols[0] if rating_cols else None

    LOGGER.info("Text column: %s, Date column: %s, Rating column: %s", TEXT, DATE, RATING)

    # basic cleaning
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

    tqdm.pandas()
    df["lemma"] = df["clean_text"].progress_apply(lambda t: lem_text(str(t)))

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
    try:
        q_low, q_high = df["review_len"].quantile([0.01, 0.99])
        df["len_outlier"] = ((df["review_len"] < q_low) | (df["review_len"] > q_high))
    except Exception:
        df["len_outlier"] = False

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
            if "df" in locals() and isinstance(df, pd.DataFrame):
                df.to_csv(OUT_CSV_PATH, index=False)
                LOGGER.info("Saved intermediate CSV to %s after failure.", OUT_CSV_PATH)
        except Exception as e:
            LOGGER.error("Failed to save intermediate CSV after exception: %s", e)
        sys.exit(1)
