#!/usr/bin/env python3
"""
synthetic_upi_fraud_with_injectors.py

Generates synthetic UPI transactions (CTGAN+VAE fallback) and injects guaranteed V1/V2/V3 anomalies,
then runs the validators and outputs a CSV.

Now UPDATED to also inject + detect SIM-SWAP (ATO) fraud using sim_swap_flag, device_id, amount.

Usage:
    python synthetic_upi_fraud_with_injectors.py
"""

import uuid
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Optional libs
try:
    from ctgan import CTGAN
    CTGAN_AVAILABLE = True
except Exception:
    CTGAN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Lambda
    from tensorflow.keras.models import Model
    from sklearn.preprocessing import StandardScaler
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ---------------------------
# Utilities
# ---------------------------
def new_tx_id():
    return str(uuid.uuid4())

BANKS = ["ybl","axl","oksbi","okhdfc","okicici","upi","paytm","axis","yes","kotak"]
NAMES = ["rama","sita","hema","purna","kumar","anil","geeta","ravi","sunil","sree","nilesh","mala","veena","ajay","rita"]

def gen_phone_upi():
    ph = str(random.randint(7000000000, 9999999999))
    return f"{ph}@{random.choice(BANKS)}"

def gen_name_upi():
    name = random.choice(NAMES) + ''.join(random.choices("abcdefghijklmnopqrstuvwxyz", k=2))
    return f"{name}@{random.choice(BANKS)}"

def gen_upi():
    return gen_phone_upi() if random.random() < 0.7 else gen_name_upi()

# Monotonic timestamp generator (global)
GLOBAL_TIMESTAMP = datetime.utcnow() - timedelta(hours=1)
def next_timestamp(min_ms=10, max_ms=1200):
    global GLOBAL_TIMESTAMP
    delta_ms = random.randint(min_ms, max_ms)
    GLOBAL_TIMESTAMP = GLOBAL_TIMESTAMP + timedelta(milliseconds=delta_ms)
    return GLOBAL_TIMESTAMP

def enforce_monotonic_unique_timestamps(df: pd.DataFrame, min_ms=5, max_ms=1200) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    if len(df)==0:
        return df
    current = pd.to_datetime(df.loc[0, "timestamp"])
    df.loc[0, "timestamp"] = current
    for i in range(1, len(df)):
        t = pd.to_datetime(df.loc[i, "timestamp"])
        if t <= current:
            t = current + timedelta(milliseconds=random.randint(min_ms, max_ms))
        current = t
        df.loc[i, "timestamp"] = t
    return df

# ---------------------------
# Schema fixer
# ---------------------------
REQUIRED_UPI_COLUMNS = [
    "tx_id",
    "from_account_id", "from_account_age_days",
    "to_account_id", "to_account_age_days",
    "timestamp", "amount",
    "velocity_in_last_hour", "received_velocity_last_hour", "pass_through_percent",
    "V1", "V2", "V3",
    "triggered_validations", "num_validations", "classification",
    "is_fraud", "fraud_type",
    # --- SIM-SWAP ADD: extra columns for SIM fraud ---
    "sim_swap_flag", "device_id", "sim_swap_detected"
]

def fix_upi_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    if "tx_id" not in df.columns:
        df["tx_id"] = [new_tx_id() for _ in range(n)]
    if "from_account_id" not in df.columns:
        df["from_account_id"] = [gen_upi() for _ in range(n)]
    if "to_account_id" not in df.columns:
        df["to_account_id"] = [gen_upi() for _ in range(n)]
    if "timestamp" not in df.columns:
        df["timestamp"] = [next_timestamp() for _ in range(n)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").fillna(datetime.utcnow())
    if "amount" not in df.columns:
        df["amount"] = np.round(np.random.uniform(5.0, 20000.0, size=n), 2)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if "from_account_age_days" not in df.columns:
        df["from_account_age_days"] = np.random.randint(1, 2000, size=n)
    if "to_account_age_days" not in df.columns:
        df["to_account_age_days"] = np.random.randint(1, 2000, size=n)

    # --- SIM-SWAP ADD: defaults for SIM fields + device_id ---
    if "sim_swap_flag" not in df.columns:
        df["sim_swap_flag"] = 0
    df["sim_swap_flag"] = df["sim_swap_flag"].fillna(0).astype(int)

    if "device_id" not in df.columns:
        df["device_id"] = [f"dev_{random.randint(100000, 999999)}" for _ in range(n)]
    df["device_id"] = df["device_id"].astype(str)

    if "sim_swap_detected" not in df.columns:
        df["sim_swap_detected"] = 0
    df["sim_swap_detected"] = df["sim_swap_detected"].fillna(0).astype(int)
    # --- END SIM-SWAP ADD ---

    for c in ["velocity_in_last_hour", "received_velocity_last_hour", "pass_through_percent"]:
        if c not in df.columns:
            df[c] = np.nan
    for c in ["V1", "V2", "V3"]:
        if c not in df.columns:
            df[c] = False
    if "triggered_validations" not in df.columns:
        df["triggered_validations"] = None
    if "num_validations" not in df.columns:
        df["num_validations"] = 0
    if "classification" not in df.columns:
        df["classification"] = "normal"
    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0
    if "fraud_type" not in df.columns:
        df["fraud_type"] = ""
    return df

# ---------------------------
# Base sample builder
# ---------------------------
def build_base_sample_upi(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    for _ in range(n):
        f = gen_upi()
        t = gen_upi()
        ts = next_timestamp()
        amt = round(random.uniform(5, 5000), 2)
        rows.append({
            "tx_id": new_tx_id(),
            "from_account_id": f,
            "to_account_id": t,
            "timestamp": ts,
            "amount": amt,
            "from_account_age_days": random.randint(1, 2000),
            "to_account_age_days": random.randint(1, 2000),
            "is_fraud": 0,
            "fraud_type": ""
        })
    return fix_upi_columns(pd.DataFrame(rows))

# ---------------------------
# CTGAN / fallback generator (UPI)
# ---------------------------
def train_ctgan_fast_upi(base_df: pd.DataFrame, epochs: int = 1):
    if not CTGAN_AVAILABLE:
        logging.info("CTGAN not available, skipping.")
        return None, ["timestamp_int", "amount", "from_account_id", "to_account_id"]
    df = base_df.copy()
    df["timestamp_int"] = df["timestamp"].astype("int64") // 10**9
    train_cols = ["timestamp_int", "amount", "from_account_id", "to_account_id"]
    ctgan = CTGAN(epochs=max(1, epochs), batch_size=500, verbose=False)
    ctgan.fit(df[train_cols], categorical_cols=["from_account_id", "to_account_id"])
    return ctgan, train_cols

def ctgan_generate_fast_upi(ctgan, train_cols, n:int, base_df:pd.DataFrame) -> pd.DataFrame:
    if ctgan is not None:
        synth = ctgan.sample(n)
        if "timestamp_int" in synth.columns:
            synth["timestamp"] = pd.to_datetime(synth["timestamp_int"], unit="s", errors="coerce")
            synth.drop(columns=["timestamp_int"], inplace=True, errors="ignore")
    else:
        logging.info("Fallback sampling: resample base with jitter")
        synth = base_df.sample(n, replace=True).reset_index(drop=True).copy()
        synth["timestamp"] = [next_timestamp() for _ in range(len(synth))]
        synth["amount"] = (synth["amount"] * np.random.uniform(0.85, 1.15, size=len(synth))).round(2)
    synth = fix_upi_columns(synth)
    return synth

# ---------------------------
# Tiny VAE fallback (numeric only)
# ---------------------------
def build_train_vae_fast_upi(base_df: pd.DataFrame, latent_dim=4, epochs=1):
    if not TF_AVAILABLE:
        logging.info("TF not available, skipping VAE.")
        return None, None, None, None, None
    df = base_df.copy()
    df["timestamp_int"] = df["timestamp"].astype("int64") // 10**9
    numeric_cols = [c for c in ["timestamp_int", "amount"] if c in df.columns]
    X = df[numeric_cols].astype(float).fillna(0.0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    input_dim = Xs.shape[1]
    inputs = Input(shape=(input_dim,))
    h = Dense(32, activation="relu")(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    def sampling(args):
        zm, zv = args
        epsilon = tf.random.normal(shape=(tf.shape(zm)[0], latent_dim))
        return zm + tf.exp(0.5 * zv) * epsilon
    z = Lambda(sampling)([z_mean, z_log_var])
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(32, activation="relu")(latent_inputs)
    outputs = Dense(input_dim, activation="linear")(x)
    decoder = Model(latent_inputs, outputs)
    outputs_decoded = decoder(z)
    class VAELossLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            x_in, x_out, zm, zv = inputs
            recon = tf.reduce_mean(tf.square(x_in - x_out))
            kl = -0.5 * tf.reduce_mean(1 + zv - tf.square(zm) - tf.exp(zv))
            self.add_loss(recon + kl)
            return x_out
    vae_outputs = VAELossLayer()([inputs, outputs_decoded, z_mean, z_log_var])
    vae = Model(inputs, vae_outputs)
    vae.compile(optimizer="adam")
    vae.fit(Xs, epochs=max(1, epochs), batch_size=64, verbose=0)
    encoder = Model(inputs, [z_mean, z_log_var, z])
    return vae, encoder, decoder, scaler, numeric_cols

def vae_generate_fast_upi(encoder, decoder, scaler, numeric_cols, n=500, latent_dim=4):
    if encoder is None:
        return pd.DataFrame()
    z_samples = np.random.normal(size=(n, latent_dim))
    decoded = decoder.predict(z_samples, verbose=0)
    decoded_inv = scaler.inverse_transform(decoded)
    df_num = pd.DataFrame(decoded_inv, columns=numeric_cols)
    if "timestamp_int" in df_num.columns:
        df_num["timestamp"] = pd.to_datetime(df_num["timestamp_int"], unit="s", errors="coerce")
        df_num.drop(columns=["timestamp_int"], inplace=True, errors="ignore")
    return df_num

# ---------------------------
# Combined generator
# ---------------------------
def generate_combined_upi(base_df: pd.DataFrame, ctgan_epochs=1, ctgan_n=1500, vae_epochs=1, vae_n=500):
    """
    Generate synthetic UPI data using:
      - CTGAN (if available; else resample+jitter)
      - Tiny VAE (if TF available; else skipped)
    Then combine both.
    """
    base_fixed = fix_upi_columns(base_df)

    # --- CTGAN part ---
    try:
        ctgan_obj, train_cols = train_ctgan_fast_upi(base_fixed, epochs=ctgan_epochs)
        synth_ctgan = ctgan_generate_fast_upi(ctgan_obj, train_cols, n=ctgan_n, base_df=base_fixed)
    except Exception:
        logging.exception("CTGAN sampling failed, falling back to resample+jitter.")
        synth_ctgan = base_fixed.sample(ctgan_n, replace=True).reset_index(drop=True).copy()
        synth_ctgan["timestamp"] = [next_timestamp() for _ in range(len(synth_ctgan))]
        synth_ctgan["amount"] = (synth_ctgan["amount"] * np.random.uniform(0.85, 1.15, size=len(synth_ctgan))).round(2)
        synth_ctgan = fix_upi_columns(synth_ctgan)

    # --- VAE part ---
    synth_vae = pd.DataFrame()
    try:
        vae_model, vae_enc, vae_dec, vae_scaler, vae_numeric_cols = build_train_vae_fast_upi(
            base_fixed, latent_dim=4, epochs=vae_epochs
        )
        if vae_enc is not None:
            df_num = vae_generate_fast_upi(vae_enc, vae_dec, vae_scaler, vae_numeric_cols, n=vae_n, latent_dim=4)
            rows = []
            for i in range(len(df_num)):
                base_row = base_fixed.sample(1).iloc[0].to_dict()
                base_row.update(df_num.iloc[i].to_dict())
                base_row["tx_id"] = new_tx_id()
                rows.append(base_row)
            synth_vae = fix_upi_columns(pd.DataFrame(rows))
    except Exception:
        logging.exception("VAE generation failed, skipping VAE part.")
        synth_vae = pd.DataFrame()

    # --- Combine ---
    if not synth_vae.empty:
        combined = pd.concat([synth_ctgan, synth_vae], ignore_index=True)
    else:
        combined = synth_ctgan

    combined = fix_upi_columns(combined)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce").fillna(datetime.utcnow())
    return combined

# ---------------------------
# Validators V1 / V2 / V3
# ---------------------------
V1_THRESH = {"unique_senders": 30, "time_window_minutes": 120, "account_age_days_max":90}
V2_THRESH = {"pass_through_percent":80, "time_window_minutes":5}
V3_THRESH = {"dormant_tx_per_day_max":5, "active_tx_per_hour_min":50}

def compute_velocity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    df["velocity_in_last_hour"] = 0
    df["received_velocity_last_hour"] = 0
    df["pass_through_percent"] = np.nan
    for idx, row in df.iterrows():
        fa, ta, ts = row["from_account_id"], row["to_account_id"], row["timestamp"]
        sent_mask = (df["from_account_id"]==fa) & (df["timestamp"] >= ts - pd.Timedelta(hours=1)) & (df["timestamp"] < ts)
        df.at[idx, "velocity_in_last_hour"] = int(sent_mask.sum())
        recv_mask = (df["to_account_id"]==ta) & (df["timestamp"] >= ts - pd.Timedelta(hours=1)) & (df["timestamp"] < ts)
        df.at[idx, "received_velocity_last_hour"] = int(recv_mask.sum())
        window_start = ts - pd.Timedelta(minutes=V2_THRESH["time_window_minutes"])
        recv_to_fa = (df["to_account_id"] == fa) & (df["timestamp"] >= window_start) & (df["timestamp"] < ts)
        rec_sum = df.loc[recv_to_fa, "amount"].sum()
        if rec_sum > 0:
            df.at[idx, "pass_through_percent"] = round((row["amount"]/rec_sum)*100, 2)
    return df

def detect_V1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df["V1"] = False
    for acc, g in df.groupby("to_account_id"):
        g_sorted = g.sort_values("timestamp").reset_index()
        times, senders = list(g_sorted["timestamp"]), list(g_sorted["from_account_id"])
        n = len(g_sorted); i=0; j=0
        while i < n:
            start = times[i]
            while j < n and (times[j] - start) <= pd.Timedelta(minutes=V1_THRESH["time_window_minutes"]):
                j += 1
            unique_senders = set(senders[i:j])
            if len(unique_senders) >= V1_THRESH["unique_senders"]:
                acc_age = int(g_sorted.loc[i, "to_account_age_days"])
                if acc_age < V1_THRESH["account_age_days_max"]:
                    idxs = g_sorted.loc[i:j-1, "index"].values
                    df.loc[idxs, "V1"] = True
            i += 1
    return df

def detect_V2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df["V2"] = False
    for idx, row in df.iterrows():
        to_acc = row["to_account_id"]; ts = row["timestamp"]
        window_end = ts + pd.Timedelta(minutes=V2_THRESH["time_window_minutes"])
        out_mask = (df["from_account_id"]==to_acc) & (df["timestamp"] > ts) & (df["timestamp"] <= window_end)
        forwarded_sum = df.loc[out_mask, "amount"].sum()
        received_amount = row["amount"]
        if received_amount > 0:
            pct = (forwarded_sum / received_amount) * 100
            if pct >= V2_THRESH["pass_through_percent"]:
                df.at[idx, "V2"] = True
                df.loc[out_mask, "V2"] = True
    return df

def detect_V3(df: pd.DataFrame, account_hist=None) -> pd.DataFrame:
    df = df.copy(); df["V3"] = False
    if account_hist is None:
        account_hist = {acc: random.choice([0,1,2,3,4,5,8,10]) for acc in df["from_account_id"].unique()}
    for acc, g in df.groupby("from_account_id"):
        hist = account_hist.get(acc, 999)
        if hist <= V3_THRESH["dormant_tx_per_day_max"]:
            times = sorted(g["timestamp"].tolist()); n=len(times); i=0; j=0
            while i < n:
                start = times[i]
                while j < n and (times[j] - start) <= pd.Timedelta(hours=1):
                    j += 1
                window_count = j - i
                if window_count >= V3_THRESH["active_tx_per_hour_min"]:
                    mask = (df["from_account_id"]==acc) & (df["timestamp"] >= start) & (df["timestamp"] <= times[j-1])
                    df.loc[mask, "V3"] = True
                i += 1
    return df

# --- SIM-SWAP CONFIG + DETECTOR ---
SIM_SWAP_CONFIG = {
    "amount_min": 40000,       # detection threshold
    "device_prefix": "new_",   # new device marker
}

def detect_sim_swap(df: pd.DataFrame, cfg=SIM_SWAP_CONFIG) -> pd.DataFrame:
    """
    Rule from your SIM-SWAP model:
      sim_swap_flag == 1
      AND device_id startswith 'new_'
      AND amount > 40000
    """
    df = df.copy()
    cond = (
        (df["sim_swap_flag"].astype(int) == 1) &
        (df["device_id"].astype(str).str.startswith(cfg["device_prefix"])) &
        (df["amount"] >= cfg["amount_min"])
    )
    df["sim_swap_detected"] = 0
    df.loc[cond, "sim_swap_detected"] = 1
    return df
# --- END SIM-SWAP DETECTOR ---

def apply_validations_and_classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["triggered_validations","num_validations","classification"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = compute_velocity_metrics(df)
    df = detect_V1(df)
    df = detect_V2(df)
    df = detect_V3(df)
    df = detect_sim_swap(df)  # --- SIM-SWAP ADD: apply SIM rule before classification ---

    def classify_row(r):
        flags = []
        if r.get("V1"): flags.append("V1")
        if r.get("V2"): flags.append("V2")
        if r.get("V3"): flags.append("V3")
        if r.get("sim_swap_detected"): flags.append("SIM_SWAP")   # --- SIM-SWAP ADD ---
        cnt = len(flags)
        cls = "upi_fraud_type" if cnt >= 1 else "normal"
        return pd.Series({
            "triggered_validations": ",".join(flags) if flags else None,
            "num_validations": cnt,
            "classification": cls
        })

    class_df = df.apply(classify_row, axis=1)
    df = pd.concat([df, class_df], axis=1)
    for c in ["V1","V2","V3"]:
        df[c] = df[c].fillna(False).astype(bool)
    df["sim_swap_detected"] = df["sim_swap_detected"].fillna(0).astype(int)
    return df

# ---------------------------
# INJECTORS (V1/V2/V3)
# ---------------------------
def inject_V1_patterns(df: pd.DataFrame, num_targets=2, unique_senders=30, window_minutes=120):
    rows = []
    for _ in range(num_targets):
        target = gen_upi()
        to_age = random.randint(1, 89)
        start = datetime.utcnow() - timedelta(hours=random.uniform(1,48))
        senders = [gen_upi() for __ in range(unique_senders)]
        times = sorted([ start + timedelta(seconds=random.randint(0, int(window_minutes*60)-1)) for __ in range(unique_senders) ])
        for s, ts in zip(senders, times):
            rows.append({
                "tx_id": new_tx_id(),
                "from_account_id": s,
                "to_account_id": target,
                "timestamp": ts,
                "amount": round(random.uniform(10,1000),2),
                "from_account_age_days": random.randint(1,2000),
                "to_account_age_days": to_age,
                "is_fraud": 1,
                "fraud_type": "V1_rapid_fan_in"
            })
    if rows:
        return fix_upi_columns(pd.DataFrame(rows))
    return pd.DataFrame()

def inject_V2_patterns(df: pd.DataFrame, num_accounts=5, incoming_per_account=6):
    rows = []
    for _ in range(num_accounts):
        acct = gen_upi()
        acct_age = random.randint(30, 1000)
        for i in range(incoming_per_account):
            sender = gen_upi()
            recv_ts = datetime.utcnow() - timedelta(hours=random.uniform(1,48)) + timedelta(seconds=random.randint(0,300))
            incoming_amount = round(random.uniform(100,5000),2)
            rows.append({
                "tx_id": new_tx_id(),
                "from_account_id": sender,
                "to_account_id": acct,
                "timestamp": recv_ts,
                "amount": incoming_amount,
                "from_account_age_days": random.randint(1,2000),
                "to_account_age_days": acct_age,
                "is_fraud": 0,
                "fraud_type": ""
            })
            forward_ts = recv_ts + timedelta(seconds=random.randint(10, 4*60+50))
            forward_amount = round(incoming_amount * random.uniform(0.8, 1.0), 2)
            recipient = gen_upi()
            rows.append({
                "tx_id": new_tx_id(),
                "from_account_id": acct,
                "to_account_id": recipient,
                "timestamp": forward_ts,
                "amount": forward_amount,
                "from_account_age_days": acct_age,
                "to_account_age_days": random.randint(1,2000),
                "is_fraud": 1,
                "fraud_type": "V2_pass_through"
            })
    return fix_upi_columns(pd.DataFrame(rows)) if rows else pd.DataFrame()

def inject_V3_patterns(df: pd.DataFrame, num_accounts=2, burst_count=60):
    rows = []
    for _ in range(num_accounts):
        acct = gen_upi()
        acct_age = random.randint(100, 2000)
        start = datetime.utcnow() - timedelta(hours=random.uniform(1,48))
        times = sorted([ start + timedelta(seconds=random.randint(0,3599)) for __ in range(burst_count) ])
        for ts in times:
            recipient = gen_upi()
            rows.append({
                "tx_id": new_tx_id(),
                "from_account_id": acct,
                "to_account_id": recipient,
                "timestamp": ts,
                "amount": round(random.uniform(5,500),2),
                "from_account_age_days": acct_age,
                "to_account_age_days": random.randint(1,2000),
                "is_fraud": 1,
                "fraud_type": "V3_spike"
            })
    return fix_upi_columns(pd.DataFrame(rows)) if rows else pd.DataFrame()

# --- SIM-SWAP INJECTOR ---
SIM_INJECT_CONFIG = {
    "num_sim_accounts": 10,
    "takeover_min_minutes": 1,
    "takeover_max_minutes": 15,
    "amount_min": 40000,
    "amount_max": 480000,
    "device_prefix": "new_",
}

def inject_sim_swap_patterns(df: pd.DataFrame, cfg=SIM_INJECT_CONFIG):
    """
    Inject SIM-SWAP ATO flows:
      - pick victim accounts
      - create high-value transfers after SIM swap
      - set sim_swap_flag=1 and device_id='new_<id>'
    """
    rows = []
    for _ in range(cfg["num_sim_accounts"]):
        victim = gen_upi()
        victim_age = random.randint(1, 30)
        sim_swap_ts = datetime.utcnow() - timedelta(hours=random.uniform(0.5,48))
        takeover_ts = sim_swap_ts + timedelta(
            minutes=random.randint(cfg["takeover_min_minutes"], cfg["takeover_max_minutes"]),
            seconds=random.randint(0,59)
        )
        amount = random.randint(cfg["amount_min"], cfg["amount_max"])
        mule = f"ACC9{random.randint(1000,999999)}@{random.choice(BANKS)}"
        rows.append({
            "tx_id": new_tx_id(),
            "from_account_id": victim,
            "to_account_id": mule,
            "timestamp": takeover_ts,
            "amount": amount,
            "from_account_age_days": victim_age,
            "to_account_age_days": random.randint(1,2000),
            "is_fraud": 1,
            "fraud_type": "SIM_SWAP_ATO",
            "sim_swap_flag": 1,
            "device_id": f"{cfg['device_prefix']}{uuid.uuid4().hex[:8]}"
        })
    return fix_upi_columns(pd.DataFrame(rows)) if rows else pd.DataFrame()
# --- END SIM-SWAP INJECTOR ---

# ---------------------------
# Apply fraud patterns (your earlier injectors kept if needed)
# ---------------------------
def inject_other_patterns(df: pd.DataFrame, patterns: list):
    # placeholder: you may also call inject_international_mix / inject_spending_spike etc.
    return pd.DataFrame()

# ---------------------------
# Pipeline: generate + inject + validate
# ---------------------------
def run_pipeline(output_csv="upi_with_injected_anomalies.csv", synthetic_count=5000, seed=42):
    random.seed(seed); np.random.seed(seed)

    # 1) Base + synthetic generation
    logging.info("Building base sample...")
    base = build_base_sample_upi(n=2000, seed=seed)

    logging.info("Generating combined synthetic (CTGAN+VAE fallback)...")
    combined = generate_combined_upi(
        base,
        ctgan_epochs=1,
        ctgan_n=synthetic_count - 500,  # keep some for VAE
        vae_epochs=1,
        vae_n=500
    )
    logging.info("Combined synthetic rows before injection: %d", len(combined))

    # 2) Inject guaranteed anomalies
    logging.info("Injecting V1 (Rapid Fan-In) patterns...")
    v1_rows = inject_V1_patterns(combined, num_targets=2, unique_senders=30, window_minutes=120)
    logging.info("V1 injected rows: %d", len(v1_rows))

    logging.info("Injecting V2 (High Pass-Through) patterns...")
    v2_rows = inject_V2_patterns(combined, num_accounts=8, incoming_per_account=6)
    logging.info("V2 injected rows: %d", len(v2_rows))

    logging.info("Injecting V3 (Dormant -> Burst) patterns...")
    v3_rows = inject_V3_patterns(combined, num_accounts=3, burst_count=60)
    logging.info("V3 injected rows: %d", len(v3_rows))

    logging.info("Injecting SIM-SWAP (ATO) patterns...")
    sim_rows = inject_sim_swap_patterns(combined)
    logging.info("SIM-SWAP injected rows: %d", len(sim_rows))

    extras = pd.concat(
        [df for df in [v1_rows, v2_rows, v3_rows, sim_rows] if not df.empty],
        ignore_index=True
    ) if (not v1_rows.empty or not v2_rows.empty or not v3_rows.empty or not sim_rows.empty) else pd.DataFrame()

    merged = pd.concat([combined, extras], ignore_index=True)
    merged = fix_upi_columns(merged)

    # 3) Make timestamps strictly increasing & unique
    merged = enforce_monotonic_unique_timestamps(merged)

    # 4) Run validations & classification (+ SIM-SWAP detection)
    final = apply_validations_and_classify(merged)
    final = final.sort_values("timestamp").reset_index(drop=True)

    # 5) Save and summary
    final.to_csv(output_csv, index=False)
    logging.info("Saved final output to %s (rows: %d)", output_csv, len(final))

    print("Total generated:", len(final))
    print("Classification distribution:")
    print(final["classification"].value_counts(dropna=False))
    print("\nFraud type counts (is_fraud=1):")
    print(final[final["is_fraud"] == 1]["fraud_type"].value_counts())

    print("\nSample anomalies (classification != 'normal'):")
    print(final[final["classification"] != "normal"].head(15).to_string(index=False))

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    run_pipeline(output_csv="upi_with_injected_anomalies.csv", synthetic_count=5000, seed=42)
