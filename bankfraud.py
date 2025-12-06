"""
synthetic_bank_fraud_final.py

Hackathon-ready synthetic UPI + Bank transfer dataset generator (Option C - MIX).
Generation strategy:
 - Layer 1 (Bayesian / Statistical): sample per-field distributions (amount, currency, merchant category)
 - Layer 2 (Markov-chain sequence simulator): generate per-account transaction sequences (idle -> low -> med -> high spend states)
This combination gives realistic per-account time-series behavior without heavy models.

Includes three fraud injectors (stubbers):
 1) smurfing (transaction structuring)
 2) money_mule cycles
 3) round_tripping (multi-hop return)

Usage: python synthetic_bank_fraud_final.py
Outputs: mixed_bank_upi_with_frauds.csv

Notes:
 - Fast: uses numpy + pandas only (no TensorFlow / CTGAN).
 - Default generates 3000 synthetic transactions (configurable).
"""

import uuid, random, math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# ------------------------------
# Utilities
# ------------------------------

def new_tx_id():
    return str(uuid.uuid4())

def new_account_id():
    return "acct_" + uuid.uuid4().hex[:10]

def new_bank_id():
    return "bank_" + uuid.uuid4().hex[:6]

def pick_random(lst):
    return random.choice(lst) if lst else None

# ------------------------------
# Base catalogs (simple realistic choices)
# ------------------------------

CURRENCIES = ["INR", "USD", "AED", "GBP", "SGD"]
MERCHANT_CATEGORIES = ["grocery", "utilities", "food_delivery", "travel", "salary", "entertainment", "ecommerce", "bank_transfer"]
MERCHANTS = [f"Merchant_{i}" for i in range(1,101)]
COUNTRIES = ["IN", "US", "GB", "SG", "AE"]

# ------------------------------
# Field-level Bayesian/statistical samplers (Layer 1)
# ------------------------------

def sample_amount(transaction_type="upi"):
    """
    Amount distribution:
     - UPI/payments: many small values (lognormal with low sigma)
     - Bank transfers: wider distribution (bigger mean + sigma)
    """
    if transaction_type == "upi":
        val = np.random.lognormal(mean=5.5, sigma=0.9)
    else:
        val = np.random.lognormal(mean=7.0, sigma=1.1)
    val = max(1.0, min(val, 5e6))
    return round(float(val), 2)

def sample_currency(country):
    if country == "IN": return "INR"
    if country == "US": return "USD"
    if country == "GB": return "GBP"
    if country == "AE": return "AED"
    if country == "SG": return "SGD"
    return pick_random(CURRENCIES)

def sample_merchant_category():
    return pick_random(MERCHANT_CATEGORIES)

def sample_merchant_name():
    return pick_random(MERCHANTS)

# ------------------------------
# Markov Chain for per-account state sequences (Layer 2)
# ------------------------------
STATES = ["IDLE", "LOW", "MED", "HIGH"]
TRANSITION_MATRIX = {
    "IDLE": {"IDLE": 0.85, "LOW": 0.12, "MED": 0.02, "HIGH": 0.01},
    "LOW":  {"IDLE": 0.20, "LOW": 0.70, "MED": 0.08, "HIGH": 0.02},
    "MED":  {"IDLE": 0.10, "LOW": 0.25, "MED": 0.55, "HIGH": 0.10},
    "HIGH": {"IDLE": 0.05, "LOW": 0.15, "MED": 0.40, "HIGH": 0.40}
}

def next_state(curr_state):
    probs = TRANSITION_MATRIX[curr_state]
    choices, weights = zip(*probs.items())
    return random.choices(choices, weights=weights, k=1)[0]

def state_time_gap_seconds(state):
    if state == "IDLE":
        return int(np.random.exponential(scale=6*3600))
    if state == "LOW":
        return int(np.random.exponential(scale=3600))
    if state == "MED":
        return int(np.random.exponential(scale=900))
    if state == "HIGH":
        return int(np.random.exponential(scale=60))
    return int(np.random.exponential(scale=3600))

def state_amount_multiplier(state):
    if state == "IDLE": return 0.3
    if state == "LOW": return 0.8
    if state == "MED": return 1.5
    if state == "HIGH": return 4.0
    return 1.0

# ------------------------------
# Generate per-account time-series using layered approach
# ------------------------------

def generate_account_sequence(account_id, n_events=100, start_time=None):
    if start_time is None:
        start_time = datetime.utcnow() - timedelta(days=30*random.random())
    events = []
    state = "IDLE" if random.random() < 0.8 else pick_random(STATES)
    ts = start_time
    for i in range(n_events):
        state = next_state(state)
        gap = state_time_gap_seconds(state)
        ts = ts + timedelta(seconds=gap)
        tx_type = "upi" if random.random() < 0.7 else "bank"
        country = pick_random(COUNTRIES)
        currency = sample_currency(country)
        base_amount = sample_amount(tx_type)
        amt = max(1.0, round(base_amount * state_amount_multiplier(state), 2))
        merchant_category = sample_merchant_category()
        merchant_name = sample_merchant_name()
        device_id = "dev_" + uuid.uuid4().hex[:8]
        bank_id = new_bank_id() if tx_type == "bank" else None
        to_account = new_account_id() if random.random() < 0.05 else None
        event = {
            "transaction_id": new_tx_id(),
            "account_id": account_id,
            "counterparty_account": to_account,
            "timestamp": ts,
            "amount": amt,
            "currency": currency,
            "merchant_country": country,
            "merchant_name": merchant_name,
            "merchant_category": merchant_category,
            "device_id": device_id,
            "bank_id": bank_id,
            "transaction_type": tx_type,
            "is_fraud": 0,
            "fraud_type": "",
            "original_tx_id": None
        }
        events.append(event)
    events = sorted(events, key=lambda x: x["timestamp"])
    return events

# ------------------------------
# Top-level generator: many accounts -> flattened dataframe
# ------------------------------

def generate_synthetic_bank_upi(num_accounts=200, events_per_account=15):
    rows = []
    for _ in range(num_accounts):
        acct = new_account_id()
        seq_len = max(1, int(np.random.poisson(events_per_account)))
        seq = generate_account_sequence(acct, n_events=seq_len)
        rows.extend(seq)
    df = pd.DataFrame(rows)
    df = fix_columns_bank_upi(df)
    df = clean_no_nan_bank_upi(df)
    return df

# ------------------------------
# Column fixer and hygiene specific to this dataset
# ------------------------------

REQUIRED_COLS_BANK = [
    "transaction_id", "account_id", "counterparty_account", "timestamp", "amount",
    "currency", "merchant_country", "merchant_name", "merchant_category",
    "device_id", "bank_id", "transaction_type", "is_fraud", "fraud_type", "original_tx_id"
]

def fix_columns_bank_upi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    if "transaction_id" not in df.columns:
        df["transaction_id"] = [new_tx_id() for _ in range(n)]
    if "account_id" not in df.columns:
        df["account_id"] = [new_account_id() for _ in range(n)]
    if "timestamp" not in df.columns:
        now = datetime.utcnow()
        df["timestamp"] = [now - timedelta(days=random.random()*30) for _ in range(n)]
    if "amount" not in df.columns:
        df["amount"] = np.round(np.random.uniform(10,2000,size=n),2)
    if "currency" not in df.columns:
        df["currency"] = "INR"
    if "merchant_country" not in df.columns:
        df["merchant_country"] = "IN"
    if "merchant_name" not in df.columns:
        df["merchant_name"] = [pick_random(MERCHANTS) for _ in range(n)]
    if "merchant_category" not in df.columns:
        df["merchant_category"] = ["misc" for _ in range(n)]
    if "device_id" not in df.columns:
        df["device_id"] = ["dev_" + uuid.uuid4().hex[:8] for _ in range(n)]
    if "bank_id" not in df.columns:
        df["bank_id"] = None
    if "transaction_type" not in df.columns:
        df["transaction_type"] = "upi"
    if "counterparty_account" not in df.columns:
        df["counterparty_account"] = None
    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0
    if "fraud_type" not in df.columns:
        df["fraud_type"] = ""
    if "original_tx_id" not in df.columns:
        df["original_tx_id"] = None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in REQUIRED_COLS_BANK:
        if col not in df.columns:
            df[col] = None
    return df

def clean_no_nan_bank_upi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    default_fill = {
        "transaction_id": "unknown_tx",
        "account_id": "unknown_acct",
        "counterparty_account": "none",
        "timestamp": pd.Timestamp.utcnow(),
        "amount": 0.0,
        "currency": "INR",
        "merchant_country": "IN",
        "merchant_name": "Unknown Merchant",
        "merchant_category": "misc",
        "device_id": "dev_unknown",
        "bank_id": "none",
        "transaction_type": "upi",
        "is_fraud": 0,
        "fraud_type": "normal",
        "original_tx_id": "none"
    }
    for col, default in default_fill.items():
        if col not in df.columns:
            df[col] = default
    df = df.fillna(default_fill)
    for col, default in default_fill.items():
        if df[col].dtype == object:
            df[col] = df[col].replace("", default)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(pd.Timestamp.utcnow())
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["is_fraud"] = df["is_fraud"].astype(int)
    return df

# ------------------------------
# Fraud injectors for Bank/UPI dataset (smurfing, money_mule, round_tripping)
# ------------------------------

def inject_smurfing(df: pd.DataFrame, cond: dict) -> pd.DataFrame:
    total_amount = float(cond.get("total_amount", 500000))
    min_tx = int(cond.get("min_tx", 8))
    max_tx = int(cond.get("max_tx", 25))
    time_window_hours = float(cond.get("time_window_hours", 2))
    beneficiary = cond.get("beneficiary")
    n = random.randint(min_tx, max_tx)
    parts = np.random.dirichlet(np.ones(n)) * total_amount
    if beneficiary:
        victim_acct = beneficiary
    else:
        victim_acct = df.sample(1).iloc[0]["account_id"]
    base_time = datetime.utcnow() - timedelta(days=random.random()*2)
    rows = []
    for amt in parts:
        src = df.sample(1).iloc[0].to_dict()
        tx_time = base_time + timedelta(seconds=random.randint(0, int(time_window_hours*3600)))
        row = {
            "transaction_id": new_tx_id(),
            "account_id": src["account_id"],
            "counterparty_account": victim_acct,
            "timestamp": tx_time,
            "amount": round(float(amt),2),
            "currency": src.get("currency","INR"),
            "merchant_country": src.get("merchant_country","IN"),
            "merchant_name": "Deposit_to_"+str(victim_acct)[:8],
            "merchant_category": "bank_transfer",
            "device_id": src.get("device_id","dev_unknown"),
            "bank_id": src.get("bank_id", None),
            "transaction_type": "bank",
            "is_fraud": 1,
            "fraud_type": "smurfing",
            "original_tx_id": None
        }
        rows.append(row)
    return pd.DataFrame(rows)

def inject_money_mule(df: pd.DataFrame, cond: dict) -> pd.DataFrame:
    deposit_amount = float(cond.get("deposit_amount", 80000))
    forward_amount = cond.get("forward_amount", deposit_amount * 0.95)
    cycle_time_min = int(cond.get("cycle_time_min", 30))
    mule_acct = df.sample(1).iloc[0]["account_id"]
    sources = df.sample(n=min(3, max(1, int(len(df)/50))), replace=True).to_dict(orient="records")
    base_time = datetime.utcnow() - timedelta(days=random.random()*2)
    rows = []
    per_source = deposit_amount / max(1, len(sources))
    for s in sources:
        row = {
            "transaction_id": new_tx_id(),
            "account_id": s["account_id"],
            "counterparty_account": mule_acct,
            "timestamp": base_time + timedelta(seconds=random.randint(0,300)),
            "amount": round(float(per_source),2),
            "currency": s.get("currency","INR"),
            "merchant_country": s.get("merchant_country","IN"),
            "merchant_name": "Deposit_to_mule",
            "merchant_category": "bank_transfer",
            "device_id": s.get("device_id","dev_unknown"),
            "bank_id": s.get("bank_id", None),
            "transaction_type": "bank",
            "is_fraud": 1,
            "fraud_type": "money_mule_deposit",
            "original_tx_id": None
        }
        rows.append(row)
    forward_to = [new_account_id() for _ in range( max(1, int(len(sources))) )]
    for tgt in forward_to:
        row = {
            "transaction_id": new_tx_id(),
            "account_id": mule_acct,
            "counterparty_account": tgt,
            "timestamp": base_time + timedelta(minutes=random.randint(1, cycle_time_min)),
            "amount": round(float(forward_amount)/len(forward_to),2),
            "currency": "INR",
            "merchant_country": "IN",
            "merchant_name": "Forward_from_mule",
            "merchant_category": "bank_transfer",
            "device_id": "dev_"+uuid.uuid4().hex[:8],
            "bank_id": None,
            "transaction_type": "bank",
            "is_fraud": 1,
            "fraud_type": "money_mule_forward",
            "original_tx_id": None
        }
        rows.append(row)
    return pd.DataFrame(rows)

def inject_round_tripping(df: pd.DataFrame, cond: dict) -> pd.DataFrame:
    hops = int(cond.get("hops", 3))
    amount = float(cond.get("amount", 200000))
    base_time = datetime.utcnow() - timedelta(days=random.random()*2)
    start_acct = df.sample(1).iloc[0]["account_id"]
    chain = [start_acct] + [ new_account_id() for _ in range(hops-1) ]
    rows = []
    for i in range(len(chain)):
        src = chain[i]
        tgt = chain[(i+1) % len(chain)]
        row = {
            "transaction_id": new_tx_id(),
            "account_id": src,
            "counterparty_account": tgt,
            "timestamp": base_time + timedelta(minutes=5*i + random.randint(0,3)),
            "amount": round(float(amount)/hops,2),
            "currency": "INR",
            "merchant_country": "IN",
            "merchant_name": f"RoundHop_{i}",
            "merchant_category": "bank_transfer",
            "device_id": "dev_"+uuid.uuid4().hex[:8],
            "bank_id": None,
            "transaction_type": "bank",
            "is_fraud": 1,
            "fraud_type": "round_tripping",
            "original_tx_id": None
        }
        rows.append(row)
    return pd.DataFrame(rows)

# ------------------------------
# Driver: apply selected patterns (simple)
# ------------------------------

def apply_fraud_patterns_bank(df: pd.DataFrame, patterns: list) -> pd.DataFrame:
    synth = df.copy().reset_index(drop=True)
    new_rows = []
    for cond in patterns:
        pattern = cond.get("pattern", "")
        count = int(cond.get("count", 1))
        for _ in range(count):
            if pattern == "smurfing":
                new_rows.append(inject_smurfing(synth, cond))
            elif pattern == "money_mule":
                new_rows.append(inject_money_mule(synth, cond))
            elif pattern == "round_tripping":
                new_rows.append(inject_round_tripping(synth, cond))
            else:
                print(f"[warn] unknown pattern: {pattern}")
    if new_rows:
        added = pd.concat(new_rows, ignore_index=True)
        added = fix_columns_bank_upi(added)
        result = pd.concat([synth, added], ignore_index=True)
    else:
        result = synth
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.sort_values("timestamp").reset_index(drop=True)
    result = clean_no_nan_bank_upi(result)
    return result

# ------------------------------
# Example usage / main demo
# ------------------------------

def main_demo(out_csv="mixed_bank_upi_with_frauds.csv"):
    print("[info] generating base synthetic bank+upi transactions (fast)...")
    base = generate_synthetic_bank_upi(num_accounts=250, events_per_account=12)
    print("[info] base rows:", len(base))

    patterns = [
        {"pattern": "smurfing", "total_amount": 300000, "min_tx": 10, "max_tx": 20, "time_window_hours": 3, "count": 3},
        {"pattern": "money_mule", "deposit_amount": 90000, "forward_amount": 85000, "cycle_time_min": 60, "count": 3},
        {"pattern": "round_tripping", "hops": 3, "amount": 200000, "count": 3}
    ]

    print("[info] injecting fraud patterns (smurfing, money_mule, round_tripping)...")
    final = apply_fraud_patterns_bank(base, patterns)
    print("[info] final rows (with frauds):", len(final))
    print(final[final["is_fraud"]==1]["fraud_type"].value_counts())
    final.to_csv(out_csv, index=False)
    print(f"[info] saved output to {out_csv}")

if __name__ == "__main__":
    main_demo()
