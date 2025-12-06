#!/usr/bin/env python3
"""
generate_upi_transactions.py
Generate 500 synthetic UPI-like transactions and mark V1/V2/V3 according to provided rules.
Outputs: transactions.csv (current working directory)
Requires: pandas, numpy
"""

import random
import string
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import argparse

# ---------------------------
# Configuration
# ---------------------------
NUM_RECORDS = 500
SEED = 42

# Validation thresholds (same as your spec)
V1_THRESH = {
    "unique_senders": 30,
    "time_window_minutes": 120,
    "account_age_days_max": 90
}
V2_THRESH = {
    "pass_through_percent": 80,
    "time_window_minutes": 5
}
V3_THRESH = {
    "dormant_tx_per_day_max": 5,
    "active_tx_per_hour_min": 50
}

# Banks and name pools for realistic UPI IDs
BANKS = ["ybl","axl","oksbi","okhdfc","okicici","upi","paytm","axis","yes","kotak"]
NAMES = ["rama","sita","hema","purna","kumar","anil","geeta","ravi","sunil","sree","nilesh","mala","veena","ajay","rita"]

# ---------------------------
# Random seeds
# ---------------------------
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# Global timestamp generator
# ---------------------------
# Start a little in the past; we'll move forward a few ms per transaction
GLOBAL_TIMESTAMP = datetime.now() - timedelta(minutes=10)

def next_timestamp():
    """
    Generate a strictly increasing timestamp by adding a small random
    number of milliseconds to the previous global timestamp.
    Ensures: all transactions have unique, realistic streaming-style timestamps.
    """
    global GLOBAL_TIMESTAMP
    delta_ms = random.randint(100, 120000)  # 10ms to 1.2s
    GLOBAL_TIMESTAMP = GLOBAL_TIMESTAMP + timedelta(milliseconds=delta_ms)
    return GLOBAL_TIMESTAMP

# ---------------------------
# Helpers for accounts
# ---------------------------
# Maintain account metadata: age (days) and historical tx/day baseline
account_meta = {}

def gen_phone_upi():
    ph = str(random.randint(7000000000, 9999999999))
    return f"{ph}@{random.choice(BANKS)}"

def gen_name_upi():
    name = random.choice(NAMES) + ''.join(random.choices(string.ascii_lowercase, k=2))
    return f"{name}@{random.choice(BANKS)}"

def gen_upi():
    # More phone-style IDs than name-style, like real UPI
    return gen_phone_upi() if random.random() < 0.65 else gen_name_upi()

def ensure_account(acc=None):
    if not acc:
        acc = gen_upi()
    if acc not in account_meta:
        # realistic age 1..2000 days
        age = random.randint(1, 2000)
        # baseline historical tx/day (more low-volume accounts)
        hist = random.choice([0,1,2,3,4,5,8,10,15])
        account_meta[acc] = {"account_age_days": age, "historical_tx_per_day": hist}
    return acc

def make_tx(from_acc, to_acc, timestamp, amount):
    return {
        "tx_id": str(uuid.uuid4())[:8],
        "from_account_id": from_acc,
        "to_account_id": to_acc,
        "amount": round(amount, 2),
        "timestamp": timestamp,
        "from_account_age_days": account_meta[from_acc]["account_age_days"],
        "to_account_age_days": account_meta[to_acc]["account_age_days"],
        "velocity_in_last_hour": None,
        "pass_through_percent": None,
        "V1": False,
        "V2": False,
        "V3": False
    }

# ---------------------------
# Build dataset with patterns
# ---------------------------
def generate_records(num_records=NUM_RECORDS):
    records = []
    pool_accounts = []

    # Pre-populate pool of random accounts
    for _ in range(220):
        a = gen_upi()
        ensure_account(a)
        pool_accounts.append(a)

    # Special pattern accounts
    # V1 target: new account (<90 days)
    v1_target = gen_upi()
    ensure_account(v1_target)
    account_meta[v1_target]["account_age_days"] = random.randint(1, V1_THRESH["account_age_days_max"] - 1)
    pool_accounts.append(v1_target)

    # V2 account: will receive and forward quickly
    v2_account = gen_upi()
    ensure_account(v2_account)
    account_meta[v2_account]["account_age_days"] = random.randint(30, 800)
    pool_accounts.append(v2_account)

    # V3 account: dormant historically, will burst
    v3_account = gen_upi()
    ensure_account(v3_account)
    account_meta[v3_account]["historical_tx_per_day"] = random.randint(0, V3_THRESH["dormant_tx_per_day_max"])
    account_meta[v3_account]["account_age_days"] = random.randint(90, 2000)
    pool_accounts.append(v3_account)

    # --- V1 pattern: >=30 unique senders -> v1_target (all within some time span)
    for _ in range(V1_THRESH["unique_senders"]):
        sender = gen_upi()
        ensure_account(sender)
        ts = next_timestamp()
        amt = random.uniform(50, 2000)
        records.append(make_tx(sender, v1_target, ts, amt))

    # --- V2 pattern: incoming txs to v2_account, many forwarded quickly
    for i in range(12):
        sender = gen_upi()
        ensure_account(sender)
        recv_ts = next_timestamp()  # incoming time
        incoming_amount = round(random.uniform(100, 5000), 2)
        incoming_tx = make_tx(sender, v2_account, recv_ts, incoming_amount)
        records.append(incoming_tx)

        # forward in ~75% cases
        if random.random() < 0.75:
            forward_ts = next_timestamp()  # guaranteed > recv_ts, small offset
            forward_amount = round(incoming_amount * random.uniform(0.8, 1.0), 2)
            recipient = gen_upi()
            ensure_account(recipient)
            records.append(make_tx(v2_account, recipient, forward_ts, forward_amount))

    # --- V3 pattern: >=50 txs in 1 hour from a dormant account
    burst_count = max(V3_THRESH["active_tx_per_hour_min"], 60)  # produce >=50, pick 60
    for _ in range(burst_count):
        ts = next_timestamp()
        recipient = gen_upi()
        ensure_account(recipient)
        amt = round(random.uniform(10, 1000), 2)
        records.append(make_tx(v3_account, recipient, ts, amt))

    # --- Fill remaining with 'normal' transactions
    remaining = num_records - len(records)
    for _ in range(remaining):
        f = random.choice(pool_accounts)
        t = random.choice(pool_accounts)
        # avoid self-send most of the time
        if f == t and random.random() < 0.9:
            t = gen_upi()
            ensure_account(t)
            pool_accounts.append(t)
        ts = next_timestamp()
        amt = round(random.uniform(10, 20000), 2)
        records.append(make_tx(f, t, ts, amt))

    return records

# ---------------------------
# Compute metrics + detection
# ---------------------------
def analyze_and_flag(records):
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # velocity_in_last_hour: count of txs from same from_account in prior 1 hour
    df['velocity_in_last_hour'] = 0
    for idx, row in df.iterrows():
        fa = row['from_account_id']
        ts = row['timestamp']
        mask = (
            (df['from_account_id'] == fa) &
            (df['timestamp'] >= ts - pd.Timedelta(hours=1)) &
            (df['timestamp'] < ts)
        )
        df.at[idx, 'velocity_in_last_hour'] = int(mask.sum())

    # received_velocity_last_hour: count of txs to same to_account in prior 1 hour
    df['received_velocity_last_hour'] = 0
    for idx, row in df.iterrows():
        ta = row['to_account_id']
        ts = row['timestamp']
        mask = (
            (df['to_account_id'] == ta) &
            (df['timestamp'] >= ts - pd.Timedelta(hours=1)) &
            (df['timestamp'] < ts)
        )
        df.at[idx, 'received_velocity_last_hour'] = int(mask.sum())

    # pass_through_percent: for an outgoing tx, if account had received funds in last 5 minutes
    df['pass_through_percent'] = np.nan
    for idx, row in df.iterrows():
        fa = row['from_account_id']
        ts = row['timestamp']
        window_start = ts - pd.Timedelta(minutes=V2_THRESH["time_window_minutes"])
        rec_mask = (
            (df['to_account_id'] == fa) &
            (df['timestamp'] >= window_start) &
            (df['timestamp'] < ts)
        )
        received_sum = df.loc[rec_mask, 'amount'].sum()
        if received_sum > 0:
            forwarded = row['amount']
            pct = (forwarded / received_sum) * 100
            df.at[idx, 'pass_through_percent'] = round(pct, 2)

    # V1 detection: sliding 2-hour window per to_account for unique senders
    df['V1'] = False
    for acc, group in df.groupby('to_account_id'):
        g = group.sort_values('timestamp').reset_index()
        times = g['timestamp'].tolist()
        senders = g['from_account_id'].tolist()
        n = len(g)
        i = 0
        j = 0
        while i < n:
            window_start = times[i]
            while j < n and (times[j] - window_start) <= pd.Timedelta(minutes=V1_THRESH["time_window_minutes"]):
                j += 1
            unique_senders = set(senders[i:j])
            if len(unique_senders) >= V1_THRESH["unique_senders"]:
                acc_age = account_meta.get(acc, {}).get('account_age_days', 9999)
                if acc_age < V1_THRESH["account_age_days_max"]:
                    idxs = g.loc[i:j-1, 'index'].values
                    df.loc[idxs, 'V1'] = True
            i += 1

    # V2 detection: for each incoming tx, check outgoing from that account in next 5 minutes
    df['V2'] = False
    for idx, row in df.iterrows():
        to_acc = row['to_account_id']
        ts = row['timestamp']
        window_end = ts + pd.Timedelta(minutes=V2_THRESH["time_window_minutes"])
        out_mask = (
            (df['from_account_id'] == to_acc) &
            (df['timestamp'] > ts) &
            (df['timestamp'] <= window_end)
        )
        forwarded_sum = df.loc[out_mask, 'amount'].sum()
        received_amount = row['amount']
        if received_amount > 0:
            pct = (forwarded_sum / received_amount) * 100
            if pct >= V2_THRESH["pass_through_percent"]:
                df.at[idx, 'V2'] = True
                df.loc[out_mask, 'V2'] = True

    # V3 detection: dormant account (<5 tx/day) suddenly >=50 tx in 1h
    df['V3'] = False
    for acc, group in df.groupby('from_account_id'):
        hist = account_meta.get(acc, {}).get('historical_tx_per_day', 999)
        if hist <= V3_THRESH["dormant_tx_per_day_max"]:
            times = group['timestamp'].sort_values().tolist()
            n = len(times)
            i = 0
            j = 0
            while i < n:
                start = times[i]
                while j < n and (times[j] - start) <= pd.Timedelta(hours=1):
                    j += 1
                window_count = j - i
                if window_count >= V3_THRESH["active_tx_per_hour_min"]:
                    mask = (
                        (df['from_account_id'] == acc) &
                        (df['timestamp'] >= start) &
                        (df['timestamp'] <= times[j-1])
                    )
                    df.loc[mask, 'V3'] = True
                i += 1

    # Classification
    def classify(row):
        flags = []
        if row['V1']: flags.append('V1')
        if row['V2']: flags.append('V2')
        if row['V3']: flags.append('V3')
        cnt = len(flags)
        if cnt >= 2:
            cls = 'upi_fraud_type'
        elif cnt == 1:
            cls = 'anomaly'
        else:
            cls = 'normal'
        return pd.Series({
            "triggered_validations": ",".join(flags) if flags else None,
            "num_validations": cnt,
            "classification": cls
        })

    class_df = df.apply(classify, axis=1)
    df = pd.concat([df, class_df], axis=1)

    # reorder columns
    cols = [
        "tx_id","timestamp",
        "from_account_id","from_account_age_days",
        "to_account_id","to_account_age_days",
        "amount",
        "velocity_in_last_hour","received_velocity_last_hour",
        "pass_through_percent",
        "V1","V2","V3",
        "triggered_validations","num_validations","classification"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].sort_values('timestamp').reset_index(drop=True)

# ---------------------------
# Main
# ---------------------------
def main(output="transactions.csv", n=NUM_RECORDS):
    print(f"Generating {n} transactions with incremental timestamps...")
    recs = generate_records(n)
    df = analyze_and_flag(recs)
    df.to_csv(output, index=False)
    print(f"Saved CSV -> {output}")
    print("Classification counts:")
    print(df['classification'].value_counts())
    print("\nSample:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic UPI transactions CSV with V1,V2,V3 flags")
    parser.add_argument("--output", "-o", default="transactions.csv", help="Output CSV filename")
    parser.add_argument("--num", "-n", type=int, default=NUM_RECORDS, help="Number of records to generate")
    args = parser.parse_args()
    main(output=args.output, n=args.num)
