#!/usr/bin/env python3
"""
clean_reviews.py

Usage (GitHub Actions will set up secrets and call this script):
  python scripts/clean_reviews.py

Expect environment variables:
  - KAGGLE_DATASET_SLUG  (owner/slug) e.g. "mamunurrahman/netflix-google-play-store-reviews"
  - KAGGLE_FILE_NAME     (optional, CSV filename inside dataset)
  - SPREADSHEET_ID       (Google Sheet ID to update)
  - SHEET_NAME           (sheet name/tab to update, default "Sheet1")
  - GCP_SA_FILE          (path to service account JSON; default '/github/home/gcp_sa.json')
  - OUT_CSV_PATH         (optional local CSV save path)
"""
import os
import sys
import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from tqdm import tqdm
import json
import logging

# gspread
import gspread
from gspread_dataframe import set_with_dataframe

# set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ========== Config from env ==========
KAGGLE_SLUG = os.getenv("KAGGLE_DATASET_SLUG", "ashishkumarak/netflix-reviews-playstore-daily-updated")  # change if desired
KAGGLE_FILE = os.getenv("Netflix_User_Behavior_Cleaned", "")  # optional specific file inside dataset
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")  # REQUIRED
SHEET_NAME = os.getenv("SHEET_NAME", "Sheet1")
GCP_SA_FILE = os.getenv("GCP_SA_FILE", "/github/home/gcp_sa.json")
OUT_CSV_PATH = os.getenv("OUT_CSV_PATH", "/github/workspace/outputs/cleaned_reviews.csv")

if not SPREADSHEET_ID:
    logging.error("SPREADSHEET_ID environment variable not set. Exiting.")
    sys.exit(2)

# create output dir
os.makedirs(os.path.dirname(OUT_CSV_PATH), exist_ok=True)

# ========== Helpers ==========
nlp = None
analyzer = SentimentIntensityAnalyzer()

def setup_spacy():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
        except Exception as e:
            logging.info("spacy model not found, trying to download (this may increase run time).")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
    return nlp

def normalize_rating(r):
    try:
        r = float(r)
        if r > 5:
            return ((r / 100.0) * 4.0) + 1.0
        return max(1.0, min(5.0, r))
    except:
        return np.nan

def strip_non_ascii(text):
    return re.sub(r"[^\x00-\x7F]+", " ", str(text))

def basic_text_clean(text):
    if not text or pd.isna(text):
        return ""
    s = str(text)
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = strip_non_ascii(s)
    s = re.sub(r"[^a-zA-Z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def lemmatize_keep_content(text, nlp_model):
    if not text:
        return ""
    doc = nlp_model(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.lemma_.strip()]
    return " ".join(tokens)

def sentiment_scores(text):
    v = analyzer.polarity_scores(str(text))
    return v['compound'], v['pos'], v['neu'], v['neg']

def outlier_iqr(series, factor=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    return (series < low) | (series > high)

# ========== Kaggle download ==========
def download_kaggle_dataset(slug, file_name=""):
    logging.info(f"Downloading Kaggle dataset {slug} ...")
    # write kaggle.json should already be present at /github/home/.kaggle/kaggle.json by GH Action step
    # CLI: kaggle datasets download -d <slug> -p <path> --unzip
    target_dir = "/github/workspace/data"
    os.makedirs(target_dir, exist_ok=True)
    cmd = f"kaggle datasets download -d {slug} -p {target_dir} --unzip --force"
    if file_name:
        # some datasets require -f <filename> but --unzip will still list the files
        cmd = f"kaggle datasets download -d {slug} -f {file_name} -p {target_dir} --unzip --force"
    logging.info(f"Running: {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        logging.error("Kaggle download failed (exit code != 0). Check dataset slug and credentials.")
        raise RuntimeError("Kaggle download failed")
    # attempt to find a CSV in target_dir
    csvs = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith('.csv')]
    if not csvs:
        raise FileNotFoundError(f"No CSV file found in {target_dir} after download.")
    logging.info(f"Found csv: {csvs[0]}")
    return csvs[0]

# ========== Main pipeline ==========
def pipeline():
    setup_spacy()
    # download dataset
    csv_path = download_kaggle_dataset(KAGGLE_SLUG, KAGGLE_FILE)
    logging.info("Loading CSV into pandas...")
    df = pd.read_csv(csv_path, low_memory=False)
    logging.info(f"Loaded {len(df)} rows, columns: {df.columns.tolist()[:12]}")

    # auto-detect text/rating/date columns
    text_cols = [c for c in df.columns if c.lower() in ['content','review','comments','comment','text','reviewtext','body']]
    date_cols = [c for c in df.columns if c.lower() in ['at','date','reviewdate','timestamp','time','created_at','review_time']]
    rating_cols = [c for c in df.columns if c.lower() in ['score','rating','ratings','stars','ratingvalue']]

    if not text_cols:
        # fallback to first object dtype col
        text_cols = [c for c in df.columns if df[c].dtype == object][:3]

    TEXT_COL = text_cols[0] if text_cols else df.columns[0]
    DATE_COL = date_cols[0] if date_cols else None
    RATING_COL = rating_cols[0] if rating_cols else None

    logging.info(f"Using TEXT_COL={TEXT_COL}, DATE_COL={DATE_COL}, RATING_COL={RATING_COL}")

    d = df.copy()
    d['review_text_raw'] = d[TEXT_COL].fillna("").astype(str)
    d['rating_raw'] = d[RATING_COL] if RATING_COL in d.columns else np.nan
    d['rating'] = d['rating_raw'].apply(normalize_rating)
    d['review_date'] = pd.to_datetime(d[DATE_COL], errors='coerce') if DATE_COL else pd.NaT

    # text cleaning + lemmatize (batch)
    logging.info("Cleaning text - basic")
    d['clean_basic'] = d['review_text_raw'].apply(basic_text_clean)

    logging.info("Lemmatizing (this may take time for large datasets).")
    tqdm.pandas()
    d['clean_text'] = d['clean_basic'].progress_apply(lambda x: lemmatize_keep_content(x, nlp))

    # features
    d['word_count'] = d['clean_text'].apply(lambda x: len(str(x).split()))
    d['char_count'] = d['review_text_raw'].apply(lambda x: len(str(x)))
    d['has_url'] = d['review_text_raw'].str.contains('http|www', na=False).astype(int)
    d['lang'] = d['review_text_raw'].apply(lambda x: 'en' if re.search(r"[a-zA-Z]", str(x)) else 'unknown')

    # sentiment
    logging.info("Computing sentiment (VADER)")
    d[['sent_compound','sent_pos','sent_neu','sent_neg']] = d['review_text_raw'].apply(lambda x: pd.Series(sentiment_scores(x)))

    d['rating_bucket'] = pd.cut(d['rating'].fillna(0), bins=[0,2,3.5,5], labels=['low','medium','high'], include_lowest=True)
    d['wordcount_outlier'] = outlier_iqr(d['word_count'])
    d['charcount_outlier'] = outlier_iqr(d['char_count'])

    # normalize numeric features if required
    scaler = MinMaxScaler()
    d[['word_count_norm','char_count_norm','sent_compound_norm']] = scaler.fit_transform(d[['word_count','char_count','sent_compound']].fillna(0))

    # topic keyword buckets (simple)
    buckets = {
        'billing': ['bill','billing','charged','charge','payment','refund','invoice','subscription','auto-renew','cancel'],
        'streaming_quality': ['buffer','lag','loading','crash','freeze','resolution','quality','stutter','pixel','buffering','slow'],
        'login_account': ['login','signin','account','password','locked','email','profile','logout','credentials'],
        'content_library': ['not available','missing','no movies','not found','region','geo','availability'],
        'subtitles_cc': ['subtitle','subtitles','cc','captions','audio','dub','caption'],
        'price': ['price','expensive','cost','plan','cheaper','subscription fee','pay'],
        'ads': ['ad','ads','advert','advertisement','commercial','promo'],
        'recommendations': ['recommend','algo','suggestion','recommender','personalized','recommendations']
    }

    def assign_buckets(text):
        t = str(text).lower()
        hits = []
        for k, keys in buckets.items():
            for kw in keys:
                if kw in t:
                    hits.append(k)
                    break
        return ';'.join(hits) if hits else 'other'

    d['topic_buckets'] = d['clean_text'].apply(assign_buckets)

    # Save local CSV
    logging.info(f"Saving cleaned CSV to {OUT_CSV_PATH}")
    d.to_csv(OUT_CSV_PATH, index=False)

    # ========== Write to Google Sheets ==========
    logging.info("Writing to Google Sheets...")
    # Authenticate with service account JSON path
    if not os.path.exists(GCP_SA_FILE):
        logging.error(f"GCP service account JSON not found at {GCP_SA_FILE}")
        raise FileNotFoundError("GCP SA JSON missing")

    gc = gspread.service_account(filename=GCP_SA_FILE)
    sh = gc.open_by_key(SPREADSHEET_ID)
    # prepare upload df (select safe columns)
    upload_df = d[['review_date','rating','rating_bucket','sent_compound','topic_buckets','word_count','char_count','clean_text']].copy()
    # convert datetimes to ISO strings for Sheets
    upload_df['review_date'] = upload_df['review_date'].apply(lambda x: x.isoformat() if pd.notna(x) else "")
    try:
        worksheet = None
        # try to open sheet by name; create if missing
        try:
            worksheet = sh.worksheet(SHEET_NAME)
            worksheet.clear()
        except Exception:
            worksheet = sh.add_worksheet(title=SHEET_NAME, rows=str(len(upload_df)+10), cols="20")
        set_with_dataframe(worksheet, upload_df, include_index=False, include_column_header=True, resize=True)
        logging.info("Google Sheet updated successfully.")
    except Exception as e:
        logging.exception("Failed to write to Google Sheet: %s", e)
        raise

if __name__ == "__main__":
    pipeline()


