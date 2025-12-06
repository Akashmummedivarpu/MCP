"""
synthetic_cc_fraud.py

Features:
- fix_columns(df): normalize/ensure necessary columns
- BaseGenerator: simple generator wrapper (uses an existing trained generator or dummy generator)
- generate_samples: produce synthetic transactions from base df using generator.predict
- fraud injectors: international_mix, rapid_high_value, spending_spike, fraud_refund
- apply_fraud_patterns(df, patterns): apply list of condition dicts to df
"""

import uuid
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ------------------------------
# Utilities
# ------------------------------

def new_tx_id():
    return str(uuid.uuid4())

def random_card_id():
    return "card_" + uuid.uuid4().hex[:12]

def random_account_id():
    return "acct_" + uuid.uuid4().hex[:10]

def pick_merchant():
    merchants = [
        ("Amazon India", "online"),
        ("Flipkart", "online"),
        ("Starbucks", "cafe"),
        ("BigBazaar", "grocery"),
        ("Hotel ABC", "travel"),
        ("Zomato", "food_delivery"),
    ]
    return random.choice(merchants)

# ------------------------------
# Column fixer
# ------------------------------

REQUIRED_COLUMNS = [
    "transaction_id", "account_id", "card_id", "timestamp", "amount",
    "currency", "merchant_country", "merchant_name", "merchant_category",
    "device_id", "is_fraud", "fraud_type", "original_tx_id"
]

def fix_columns(df: pd.DataFrame, home_country: str = "IN") -> pd.DataFrame:
    """
    Ensure the dataframe has consistent columns needed for credit-card transactions.
    Adds defaults if missing. Converts timestamp to pandas datetime.
    """
    df = df.copy()
    # common fallback columns
    if "transaction_id" not in df.columns:
        df["transaction_id"] = [new_tx_id() for _ in range(len(df))]
    if "account_id" not in df.columns:
        df["account_id"] = [random_account_id() for _ in range(len(df))]
    if "card_id" not in df.columns:
        df["card_id"] = [random_card_id() for _ in range(len(df))]
    if "timestamp" not in df.columns:
        # generate timestamps in last 30 days
        now = datetime.utcnow()
        df["timestamp"] = [now - timedelta(days=random.random()*30) for _ in range(len(df))]
    if "amount" not in df.columns:
        df["amount"] = np.round(np.random.uniform(10, 2000, size=len(df)), 2)
    if "currency" not in df.columns:
        df["currency"] = "INR"
    if "merchant_country" not in df.columns:
        df["merchant_country"] = home_country
    if "merchant_name" not in df.columns or "merchant_category" not in df.columns:
        merch = [pick_merchant() for _ in range(len(df))]
        df["merchant_name"] = [m[0] for m in merch]
        df["merchant_category"] = [m[1] for m in merch]
    if "device_id" not in df.columns:
        df["device_id"] = ["dev_" + uuid.uuid4().hex[:8] for _ in range(len(df))]
    # fraud markers
    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0
    if "fraud_type" not in df.columns:
        df["fraud_type"] = ""
    if "original_tx_id" not in df.columns:
        df["original_tx_id"] = None

    # ensure timestamp type
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # keep only required and extra columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[ list(df.columns) ]


# ------------------------------
# Fraud injectors
# ------------------------------

def inject_international_mix(df: pd.DataFrame, cond: dict, home_country="IN") -> pd.DataFrame:
    """
    Pattern:
      - total_tx: number of transactions in the pattern
      - same_country: number of tx in home country
      - foreign_countries: number of tx in different countries (count)
      - time_window_hours: window within which all transactions happen
    """
    out = []
    total_tx = int(cond.get("total_tx", 5))
    same_country = int(cond.get("same_country", 3))
    foreign_count = int(cond.get("foreign_countries", 2))
    time_window_hours = float(cond.get("time_window_hours", 24))

    # Build pool of foreign countries (simple list)
    foreign_pool = ["US", "GB", "SG", "AE", "FR", "DE", "CN", "JP", "AU"]

    # pick a victim account from df
    acct_row = df.sample(1).iloc[0]
    acct = acct_row["account_id"]
    # choose start time
    start = acct_row["timestamp"]
    # ensure start is now-ish if older
    start = pd.to_datetime(start)
    start = datetime.utcnow() - timedelta(days=np.random.random()*2) if np.random.rand() < 0.5 else start

    # generate timestamps within window (sorted, short intervals)
    times = sorted([ start + timedelta(seconds=int(np.random.random()*time_window_hours*3600))
                     for _ in range(total_tx) ])

    # create same_country txs
    for i in range(same_country):
        row = acct_row.copy()
        row = row.to_dict()
        row["transaction_id"] = new_tx_id()
        row["account_id"] = acct
        row["timestamp"] = times.pop(0)
        row["merchant_country"] = home_country
        row["currency"] = "INR"
        row["amount"] = round(max(10.0, np.random.normal(100,50)),2)
        row["is_fraud"] = 1
        row["fraud_type"] = "international_mix"
        out.append(row)

    # foreign txs - choose distinct countries
    chosen_foreign = random.sample(foreign_pool, k=foreign_count)
    for country in chosen_foreign:
        row = acct_row.copy()
        row = row.to_dict()
        row["transaction_id"] = new_tx_id()
        row["account_id"] = acct
        row["timestamp"] = times.pop(0) if times else datetime.utcnow()
        row["merchant_country"] = country
        row["currency"] = "USD" if country != "IN" else "INR"
        row["amount"] = round(max(50.0, np.random.normal(150,80)),2)
        row["is_fraud"] = 1
        row["fraud_type"] = "international_mix"
        out.append(row)

    return pd.DataFrame(out)


def inject_rapid_high_value(df: pd.DataFrame, cond: dict) -> pd.DataFrame:
    """
    Pattern:
      - max_amount: threshold (we will create tx > max_amount)
      - tx_count: number of tx
      - time_window_min: within N minutes
    """
    tx_count = int(cond.get("tx_count", 4))
    threshold = float(cond.get("max_amount", 5000))
    time_window_min = float(cond.get("time_window_min", 10))

    acct_row = df.sample(1).iloc[0]
    acct = acct_row["account_id"]
    base_time = acct_row["timestamp"]
    base_time = datetime.utcnow() - timedelta(hours=np.random.random()*48) if np.random.rand() < 0.5 else pd.to_datetime(base_time)
    times = [ base_time + timedelta(seconds=random.randint(0, int(time_window_min*60))) for _ in range(tx_count)]
    out = []
    for t in times:
        row = acct_row.copy().to_dict()
        row["transaction_id"] = new_tx_id()
        row["account_id"] = acct
        row["timestamp"] = t
        # make them high value (random above threshold)
        row["amount"] = round(np.random.uniform(threshold * 1.1, threshold * 3.0), 2)
        row["is_fraud"] = 1
        row["fraud_type"] = "rapid_high_value"
        row["merchant_country"] = row.get("merchant_country", "IN")
        out.append(row)
    return pd.DataFrame(out)


def inject_spending_spike(df: pd.DataFrame, cond: dict) -> pd.DataFrame:
    """
    Pattern:
      If baseline_daily_avg <= X but sudden_spend happens in time_window_hours
      - baseline_daily_avg: expected baseline (if df available, we try to compute)
      - sudden_spend: total spend in the window (we'll create 1..n tx summing to this)
      - time_window_hours: hours window in which spike happens
    """
    baseline_daily_avg = float(cond.get("baseline_daily_avg", 2000))
    sudden_spend = float(cond.get("sudden_spend", 50000))
    time_window_hours = float(cond.get("time_window_hours", 1))
    # choose account - prefer one with low avg if possible
    acct = None
    try:
        avg_by_acct = df.groupby("account_id")["amount"].mean()
        low_accts = avg_by_acct[avg_by_acct <= baseline_daily_avg].index.tolist()
        if low_accts:
            acct = random.choice(low_accts)
    except Exception:
        acct = None
    if acct is None:
        acct = df.sample(1).iloc[0]["account_id"]
    acct_row = df[df["account_id"] == acct].sample(1).iloc[0]
    base_time = pd.to_datetime(acct_row["timestamp"])
    base_time = datetime.utcnow() - timedelta(days=np.random.random()*2)
    # decide number of tx to create to sum to sudden_spend (2-6 tx)
    k = random.randint(1, 6)
    parts = np.random.dirichlet(np.ones(k)) * sudden_spend
    times = sorted([ base_time + timedelta(seconds=int(np.random.random()*time_window_hours*3600))
                     for _ in range(k) ])
    out = []
    for amt, t in zip(parts, times):
        row = acct_row.copy().to_dict()
        row["transaction_id"] = new_tx_id()
        row["account_id"] = acct
        row["timestamp"] = t
        row["amount"] = round(max(1.0, float(amt)), 2)
        row["is_fraud"] = 1
        row["fraud_type"] = "spending_spike"
        out.append(row)
    return pd.DataFrame(out)


def inject_fraud_refund(df: pd.DataFrame, cond: dict) -> pd.DataFrame:
    """
    Pattern:
      create a purchase of purchase_amount and then a refund of refund_amount (>purchase_amount)
      - purchase_amount: amount of original purchase
      - refund_amount: amount of refund (we'll create refund tx referencing original_tx_id)
    """
    purchase_amount = float(cond.get("purchase_amount", 1500))
    refund_amount = float(cond.get("refund_amount", 3000))

    acct_row = df.sample(1).iloc[0]
    acct = acct_row["account_id"]
    time_purchase = pd.to_datetime(acct_row["timestamp"])
    # original purchase
    purchase_tx = acct_row.copy().to_dict()
    purchase_tx["transaction_id"] = new_tx_id()
    purchase_tx["account_id"] = acct
    purchase_tx["timestamp"] = time_purchase
    purchase_tx["amount"] = round(purchase_amount, 2)
    purchase_tx["is_fraud"] = 0   # real purchase might be legit
    purchase_tx["fraud_type"] = ""

    # refund - occurs shortly after
    refund_tx = acct_row.copy().to_dict()
    refund_tx["transaction_id"] = new_tx_id()
    refund_tx["account_id"] = acct
    refund_tx["timestamp"] = time_purchase + timedelta(minutes=random.randint(1, 180))
    refund_tx["amount"] = round(-abs(refund_amount), 2)   # refunds as negative amounts commonly
    refund_tx["is_fraud"] = 1
    refund_tx["fraud_type"] = "fraud_refund"
    refund_tx["original_tx_id"] = purchase_tx["transaction_id"]

    return pd.DataFrame([purchase_tx, refund_tx])



# ------------------------------
# Apply patterns (driver)
# ------------------------------

def apply_fraud_patterns(synthetic_df: pd.DataFrame, patterns: list, home_country="IN") -> pd.DataFrame:
    """
    patterns: list of condition dicts (as in user's JSON examples)
    returns a copy of synthetic_df with fraud rows added and flagged.
    """
    synth = synthetic_df.copy().reset_index(drop=True)
    synth["is_fraud"] = synth.get("is_fraud", 0).fillna(0).astype(int)
    synth["fraud_type"] = synth.get("fraud_type", "").fillna("")
    new_rows = []
    for cond in patterns:
        pattern = cond.get("pattern", "")
        if pattern == "international_mix":
            df_new = inject_international_mix(synth, cond, home_country=home_country)
            new_rows.append(df_new)
        elif pattern == "rapid_high_value":
            df_new = inject_rapid_high_value(synth, cond)
            new_rows.append(df_new)
        elif pattern == "spending_spike":
            df_new = inject_spending_spike(synth, cond)
            new_rows.append(df_new)
        elif pattern == "fraud_refund":
            df_new = inject_fraud_refund(synth, cond)
            new_rows.append(df_new)
        else:
            print(f"[warn] unknown pattern {pattern} - skipping")
    if new_rows:
        added = pd.concat(new_rows, ignore_index=True)
        # ensure required columns and types
        added = fix_columns(added)
        result = pd.concat([synth, added], ignore_index=True)
    else:
        result = synth
    # ensure timestamp dtype
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    # sort by timestamp for convenience
    result = result.sort_values("timestamp").reset_index(drop=True)
    return clean_no_nan(result)

def clean_no_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the final dataframe has ZERO NaN or empty values.
    Applies consistent defaults across all known fields.
    """
    df = df.copy()

    # Default fill rules
    default_fill = {
        "transaction_id": "unknown_tx",
        "account_id": "unknown_acct",
        "card_id": "unknown_card",
        "timestamp": pd.Timestamp.utcnow(),
        "amount": 0.0,
        "currency": "INR",
        "merchant_country": "IN",
        "merchant_name": "Unknown Merchant",
        "merchant_category": "misc",
        "device_id": "dev_unknown",
        "is_fraud": 0,
        "fraud_type": "normal",
        "original_tx_id": "none"
    }

    # 1. Fill missing columns with defaults
    for col, default in default_fill.items():
        if col not in df.columns:
            df[col] = default

    # 2. Fill NaN in existing columns
    df = df.fillna(default_fill)

    # 3. Clean empty strings → default
    for col, default in default_fill.items():
        if df[col].dtype == object:
            df[col] = df[col].replace("", default)

    # 4. Enforce datatypes
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(pd.Timestamp.utcnow())
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # Ensure boolean fraud
    df["is_fraud"] = df["is_fraud"].astype(int)

    return df


# ------------------------------
# CTGAN Generator
# ------------------------------

from ctgan import CTGAN

def train_ctgan(base_df: pd.DataFrame, epochs=30, batch_size=500):
    """
    Train CTGAN on the cleaned base dataframe.
    Only uses numeric/categorical columns CTGAN can handle.
    """
    # CTGAN cannot handle timestamps directly → convert to numeric (epoch)
    df = base_df.copy()

    df["timestamp_int"] = df["timestamp"].astype("int64") // 10**9

    # Columns CTGAN will model
    # Remove non-suitable fields like fraud markers
    train_cols = [
        "timestamp_int", "amount", "currency", "merchant_country",
        "merchant_name", "merchant_category", "account_id", "card_id",
        "device_id"
    ]

    train_df = df[train_cols].copy()

    # Categorical Columns
    categorical_cols = [
        "currency", "merchant_country", "merchant_name", 
        "merchant_category", "account_id", "card_id", "device_id"
    ]

    ctgan = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )
    ctgan.fit(train_df, categorical_cols)

    return ctgan, train_cols, categorical_cols


def ctgan_generate(ctgan, train_cols, n=1000):
    """
    Generate synthetic transactions using CTGAN and convert timestamp back.
    """
    synth = ctgan.sample(n)

    # Convert timestamp back to pandas datetime
    synth["timestamp"] = pd.to_datetime(synth["timestamp_int"], unit="s", errors="coerce")
    synth.drop(columns=["timestamp_int"], inplace=True)

    # Add required missing fields
    synth["transaction_id"] = [new_tx_id() for _ in range(len(synth))]
    synth["is_fraud"] = 0
    synth["fraud_type"] = ""
    synth["original_tx_id"] = None

    # Fix final column structure
    synth = fix_columns(synth)

    return synth



# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    # 1) create a sample base dataset
    sample_base = pd.DataFrame({
        "transaction_id": [new_tx_id() for _ in range(2000)],
        "account_id": [random_account_id() for _ in range(2000)],
        "card_id": [random_card_id() for _ in range(2000)],
        "timestamp": [datetime.utcnow() - timedelta(days=random.random()*60) for _ in range(2000)],
        "amount": np.round(np.random.uniform(10, 5000, size=2000), 2),
        "currency": ["INR"]*2000,
        "merchant_country": ["IN"]*2000,
        "merchant_name": ["Merchant_"+str(i % 50) for i in range(2000)],
        "merchant_category": ["misc"]*2000,
        "device_id": ["dev_"+str(i % 80) for i in range(2000)]
    })

    # 2) Fix columns before CTGAN train
    base_fixed = fix_columns(sample_base, home_country="IN")

    # 3) Train CTGAN
    ctgan, train_cols, cat_cols = train_ctgan(base_fixed, epochs=0)

    # 4) Generate 1000 rows from CTGAN
    synthetic_ctgan = ctgan_generate(ctgan, train_cols, n=1000)

    # 5) Apply fraud patterns on CTGAN-generated data
    patterns = [
        {"pattern": "international_mix", "total_tx": 5, "same_country": 3, "foreign_countries": 2, "time_window_hours": 24},
        {"pattern": "rapid_high_value", "max_amount": 5000, "tx_count": 4, "time_window_min": 10},
        {"pattern": "spending_spike", "baseline_daily_avg": 2000, "sudden_spend": 50000, "time_window_hours": 1},
        {"pattern": "fraud_refund", "purchase_amount": 1500, "refund_amount": 3000}
    ]

    final = apply_fraud_patterns(synthetic_ctgan, patterns, home_country="IN")

    # Output
    print("CTGAN Synthetic:", len(synthetic_ctgan))
    print("Final:", len(final))
    print("Fraud counts:")
    print(final[final["is_fraud"] == 1]["fraud_type"].value_counts())

    final.to_csv("ctgan_with_frauds.csv", index=False)
