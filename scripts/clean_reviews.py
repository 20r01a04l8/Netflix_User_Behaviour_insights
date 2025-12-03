#!/usr/bin/env python3
"""
clean_reviews.py

Clean Netflix Google Play review dataset, extract features, and upload to Google Sheets.

This version is adapted for GitHub Actions:
- Uses GITHUB_WORKSPACE and HOME environment variables (no hard-coded /github paths).
- Reads KAGGLE_CONFIG_DIR and GCP_SA_FILE env vars (workflow writes these).
- Writes cleaned CSV to OUT_CSV_PATH (default: <GITHUB_WORKSPACE>/outputs/cleaned_reviews.csv).

Expected environment variables (set by the workflow):
  - KAGGLE_DATASET_SLUG  (owner/slug) e.g. "mamunurrahman/netflix-google-play-store-reviews"
  - KAGGLE_FILE_NAME     (optional, CSV filename inside dataset)
  - KAGGLE_CONFIG_DIR    (path to ~/.kaggle where kaggle.json is written)
  - SPREADSHEET_ID       (Google Sheets spreadsheet id)  <-- REQUIRED
  - SHEET_NAME           (sheet/tab name, default "Sheet1")
  - GCP_SA_FILE          (path to service account JSON; workflow sets it to $HOME/gcp_sa.json)
  - OUT_CSV_PATH         (optional local CSV save path)
"""

import os
import sys
import logging
import re
import json
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

# NLP & sentiment
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# GSpread
import gspread
from gspread_dataframe import set_with_dataframe

from sklearn.preprocessing import MinMaxScaler

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# -----------------------------
# Environment and path config
# -----------------------------
GITHUB_WORKSPACE = os.getenv("GITHUB_WORKSPACE", os.path.abspath("."))
HOME_DIR = os.getenv("HOME", os.path.expanduser("~"))

KAGGLE_SLUG = os.getenv("KAGGLE_DATASET_SLUG", "ashishkumarak/netflix-reviews-playstore-daily-updated")
KAGGLE_FILE = os.getenv("KAGGLE_FILE_NAME", "")  # optional
KAGGLE_CONFIG_DIR = os.getenv("KAGGLE_CONFIG_DIR", os.path.join(HOME_DIR, ".kaggle"))

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")  # REQUIRED
SHEET_NAME = os.getenv("SHEET_NAME", "Sheet1")
GCP_SA_FILE = os.getenv("GCP_SA_FILE", os.path.join(HOME_DIR, "gcp_sa.json"))

OUT_CSV_PATH = os.getenv("OUT_CSV_PATH", os.path.join(GITHUB_WORKSPACE, "outputs", "cleaned_reviews.csv"))

# Ensure directories exist
os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)
os.makedirs(os.path.join(GITHUB_WORKSPACE, "data"), exist_ok=True)
os.makedirs(KAGGLE_CONFIG_DIR, exist_ok=True)

# -----------------------------
# Helpers: text, sentiment, NLP
# -----------------------------
nlp = None
analyzer = SentimentIntensityAnalyzer()


def setup_spacy():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except Exception:
            logging.info("spaCy model not present. Attempting to download en_core_web_sm...")
            import subprocess

            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return nlp


def normalize_rating(r):
    try:
        r = float(r)
        if r > 5:
            # heuristic: 0-100 scale -> map to 1-5
            return ((r / 100.0) * 4.0) + 1.0
        return max(1.0, min(5.0, r))
    except Exception:
        return np.nan


def strip_non_ascii(text):
    return re.sub(r"[^\x00-\x7F]+", " ", str(text))


def basic_text_clean(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    s = str(text)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = strip_non_ascii(s)
    s = re.sub(r"[^a-zA-Z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def lemmatize_keep_content(text: str, nlp_model) -> str:
    if not text:
        return ""
    doc = nlp_model(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.lemma_.strip()]
    return " ".join(tokens)


def sentiment_scores(text: str):
    v = analyzer.polarity_scores(str(text))
    return v["compound"], v["pos"], v["neu"], v["neg"]


def outlier_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    return (series < low) | (series > high)


# -----------------------------
# Kaggle download
# -----------------------------
def download_kaggle_dataset(slug: str, file_name: str = "") -> str:
    """
    Download Kaggle dataset using kaggle CLI. Expects KAGGLE_CONFIG_DIR to contain kaggle.json or workflow exported KAGGLE_CONFIG_DIR env.
    Returns path to the first CSV found in the target_dir.
    """
    logging.info("Downloading Kaggle dataset: %s", slug)
    target_dir = os.path.join(GITHUB_WORKSPACE, "data")
    os.makedirs(target_dir, exist_ok=True)

    # Prepare command. Use --unzip to extract if zipped.
    if file_name:
        cmd = f"kaggle datasets download -d {slug} -f {file_name} -p {target_dir} --unzip --force"
    else:
        cmd = f"kaggle datasets download -d {slug} -p {target_dir} --unzip --force"

    # Ensure KAGGLE_CONFIG_DIR env is set for the subprocess
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = KAGGLE_CONFIG_DIR

    logging.info("Running command: %s", cmd)
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(
            "Kaggle download failed (exit code != 0). "
            "Check KAGGLE_JSON secret and KAGGLE_DATASET_SLUG. See logs."
        )

    # find CSV
    csvs = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith(".csv")]
    if not csvs:
        # maybe the dataset produced a folder with CSVs
        for root, _, files in os.walk(target_dir):
            for f in files:
                if f.lower().endswith(".csv"):
                    csvs.append(os.path.join(root, f))
    if not csvs:
        raise FileNotFoundError(f"No CSV file found in {target_dir} after kaggle download/unzip.")
    logging.info("Found CSV: %s", csvs[0])
    return csvs[0]


# -----------------------------
# Pipeline
# -----------------------------
def pipeline():
    logging.info("Starting pipeline.")
    logging.info("Environment summary: GITHUB_WORKSPACE=%s, HOME=%s", GITHUB_WORKSPACE, HOME_DIR)
    logging.info("KAGGLE_SLUG=%s, KAGGLE_FILE=%s", KAGGLE_SLUG, KAGGLE_FILE)
    logging.info("SPREADSHEET_ID=%s, SHEET_NAME=%s", "SET" if SPREADSHEET_ID else "MISSING", SHEET_NAME)
    logging.info("GCP_SA_FILE=%s", GCP_SA_FILE)
    logging.info("OUT_CSV_PATH=%s", OUT_CSV_PATH)

    if not SPREADSHEET_ID:
        logging.error("Missing SPREADSHEET_ID environment variable. Exiting.")
        sys.exit(2)

    # Setup spaCy and sentiment
    setup_spacy()

    # Download dataset via Kaggle CLI
    try:
        csv_path = download_kaggle_dataset(KAGGLE_SLUG, KAGGLE_FILE)
    except Exception as e:
        logging.exception("Failed to download Kaggle dataset: %s", e)
        raise

    # Load CSV into pandas
    logging.info("Loading CSV into pandas: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    logging.info("Loaded rows: %d, columns: %s", len(df), df.columns.tolist()[:20])

    # Auto-detect likely text, date, rating columns
    text_cols = [c for c in df.columns if c.lower() in ["content", "review", "comments", "comment", "text", "reviewtext", "body"]]
    date_cols = [c for c in df.columns if c.lower() in ["at", "date", "reviewdate", "timestamp", "time", "created_at", "review_time"]]
    rating_cols = [c for c in df.columns if c.lower() in ["score", "rating", "ratings", "stars", "ratingvalue"]]

    if not text_cols:
        # fallback to first object dtype column(s)
        text_cols = [c for c in df.columns if df[c].dtype == object][:3]

    TEXT_COL = text_cols[0] if text_cols else df.columns[0]
    DATE_COL = date_cols[0] if date_cols else None
    RATING_COL = rating_cols[0] if rating_cols else None

    logging.info("Using TEXT_COL=%s, DATE_COL=%s, RATING_COL=%s", TEXT_COL, DATE_COL, RATING_COL)

    d = df.copy()
    d["review_text_raw"] = d[TEXT_COL].fillna("").astype(str)
    d["rating_raw"] = d[RATING_COL] if RATING_COL in d.columns else np.nan
    d["rating"] = d["rating_raw"].apply(normalize_rating)
    d["review_date"] = pd.to_datetime(d[DATE_COL], errors="coerce") if DATE_COL else pd.NaT

    # Basic text cleaning
    logging.info("Cleaning text (basic). This may take a while on large datasets.")
    d["clean_basic"] = d["review_text_raw"].apply(basic_text_clean)

    # Lemmatize with spaCy (progress)
    tqdm.pandas()
    d["clean_text"] = d["clean_basic"].progress_apply(lambda x: lemmatize_keep_content(x, nlp))

    # Features
    d["word_count"] = d["clean_text"].apply(lambda x: len(str(x).split()))
    d["char_count"] = d["review_text_raw"].apply(lambda x: len(str(x)))
    d["has_url"] = d["review_text_raw"].str.contains("http|www", na=False).astype(int)
    d["lang"] = d["review_text_raw"].apply(lambda x: "en" if re.search(r"[a-zA-Z]", str(x)) else "unknown")

    # Sentiment
    logging.info("Computing sentiment (VADER).")
    d[["sent_compound", "sent_pos", "sent_neu", "sent_neg"]] = d["review_text_raw"].apply(lambda x: pd.Series(sentiment_scores(x)))

    # Buckets and outliers
    d["rating_bucket"] = pd.cut(d["rating"].fillna(0), bins=[0, 2, 3.5, 5], labels=["low", "medium", "high"], include_lowest=True)
    d["wordcount_outlier"] = outlier_iqr(d["word_count"])
    d["charcount_outlier"] = outlier_iqr(d["char_count"])

    # Normalize numeric features
    scaler = MinMaxScaler()
    try:
        d[["word_count_norm", "char_count_norm", "sent_compound_norm"]] = scaler.fit_transform(d[["word_count", "char_count", "sent_compound"]].fillna(0))
    except Exception:
        # If something odd happens (e.g., constant column), fallback to zeros
        d["word_count_norm"] = 0
        d["char_count_norm"] = 0
        d["sent_compound_norm"] = 0

    # Topic keyword buckets (simple rule-based)
    buckets = {
        "billing": ["bill", "billing", "charged", "charge", "payment", "refund", "invoice", "subscription", "auto-renew", "cancel"],
        "streaming_quality": ["buffer", "lag", "loading", "crash", "freeze", "resolution", "quality", "stutter", "pixel", "buffering", "slow"],
        "login_account": ["login", "signin", "account", "password", "locked", "email", "profile", "logout", "credentials"],
        "content_library": ["not available", "missing", "no movies", "not found", "region", "geo", "availability"],
        "subtitles_cc": ["subtitle", "subtitles", "cc", "captions", "audio", "dub", "caption"],
        "price": ["price", "expensive", "cost", "plan", "cheaper", "subscription fee", "pay"],
        "ads": ["ad", "ads", "advert", "advertisement", "commercial", "promo"],
        "recommendations": ["recommend", "algo", "suggestion", "recommender", "personalized", "recommendations"],
    }

    def assign_buckets(text):
        t = str(text).lower()
        hits = []
        for k, keys in buckets.items():
            for kw in keys:
                if kw in t:
                    hits.append(k)
                    break
        return ";".join(hits) if hits else "other"

    d["topic_buckets"] = d["clean_text"].apply(assign_buckets)

    # Save cleaned CSV
    logging.info("Saving cleaned CSV to: %s", OUT_CSV_PATH)
    d.to_csv(OUT_CSV_PATH, index=False)

    # -----------------------------
    # Write to Google Sheets
    # -----------------------------
    logging.info("Attempting to write to Google Sheets (spreadsheet id: %s)", SPREADSHEET_ID)
    if not os.path.exists(GCP_SA_FILE):
        logging.error("GCP service account JSON not found at %s", GCP_SA_FILE)
        raise FileNotFoundError("GCP SA JSON missing - ensure GCP_SA_FILE env is set and file exists.")

    # Authenticate via service account
    try:
        gc = gspread.service_account(filename=GCP_SA_FILE)
    except Exception as e:
        logging.exception("Failed to authenticate with gspread using service account JSON: %s", e)
        raise

    try:
        sh = gc.open_by_key(SPREADSHEET_ID)
    except Exception as e:
        logging.exception("Failed to open spreadsheet by id: %s", e)
        raise

    # Prepare upload dataframe (select lightweight columns)
    upload_df = d[["review_date", "rating", "rating_bucket", "sent_compound", "topic_buckets", "word_count", "char_count", "clean_text"]].copy()
    upload_df["review_date"] = upload_df["review_date"].apply(lambda x: x.isoformat() if pd.notna(x) else "")

    try:
        worksheet = None
        try:
            worksheet = sh.worksheet(SHEET_NAME)
            worksheet.clear()
        except Exception:
            worksheet = sh.add_worksheet(title=SHEET_NAME, rows=str(len(upload_df) + 10), cols="20")
        set_with_dataframe(worksheet, upload_df, include_index=False, include_column_header=True, resize=True)
        logging.info("Google Sheet updated successfully (sheet: %s).", SHEET_NAME)
    except Exception as e:
        logging.exception("Failed to write DataFrame to Google Sheet: %s", e)
        raise

    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    pipeline()
