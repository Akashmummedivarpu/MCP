import random
import uuid
from datetime import datetime, timedelta, timezone
from collections import deque, defaultdict
import pandas as pd
import numpy as np

# Asia/Kolkata timezone
IST = timezone(timedelta(hours=5, minutes=30))

def _rand_geo():
    lat = round(random.uniform(8.0, 30.0), 6)
    lon = round(random.uniform(68.0, 97.0), 6)
    return lat, lon


def generate_and_save_transactions():

    n = 500
    inject_sim_count = 20
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    num_accounts = 200
    accounts = [f"ACC{100000 + i}" for i in range(num_accounts)]
    payees = [f"ACC{200000 + i}" for i in range(400)]
    device_pool = [f"dev_{i:06d}" for i in range(300)]
    merchant_names = ["FlipKart", "Amazon", "BigBazaar", "PayTM_Merchant",
                      "Local_Shop", "Jewellery_Mart", "ElectroPlus", "MovieTheatre"]
    categories = ["GROCERY","ELECTRONICS","JEWELLERY","UTILITY",
                  "REMITTANCE","TRAVEL","RESTURANT"]
    auth_codes = ["APPROVED","DECLINED"]
    ip_blocks = ["192.168.", "103.21.", "45.33.", "40.77.",
                 "14.139.", "49.36.", "103.198."]

    base_time = datetime.now(IST) - timedelta(minutes=30)

    timestamps = []
    current = base_time
    for _ in range(n):
        step = random.choices([1,2,3,5,10,20,60,90],
                              weights=[30,20,15,10,8,8,5,4])[0]
        current += timedelta(seconds=step)
        timestamps.append(current)

    records = []

    inject_sim_count = min(inject_sim_count, n//2)
    sim_accounts = random.sample(accounts, inject_sim_count)
    injection_positions = random.sample(range(n//2), inject_sim_count)

    injection_map = {}
    for acc, pos in zip(sim_accounts, injection_positions):
        sim_swap_ts = timestamps[max(0, pos-2)]
        takeover_ts = sim_swap_ts + timedelta(minutes=random.randint(1,7),
                                              seconds=random.randint(0,59))
        injection_map[pos] = {
            "from_account_id": acc,
            "sim_swap_ts": sim_swap_ts,
            "takeover_ts": takeover_ts,
        }

    for i in range(n):

        ts = timestamps[i]
        is_injected = False

        if i in injection_map:
            info = injection_map[i]

            from_acc = info["from_account_id"]
            ts = info["takeover_ts"]
            device_id = f"new_{uuid.uuid4().hex[:8]}"
            amount = random.randint(45000, 480000)
            transaction_type = "TRANSFER"
            auth_code = "APPROVED"
            to_acc = f"ACC{900000 + random.randint(1,9999)}"
            sim_flag = 1
            is_injected = True

        else:
            from_acc = random.choice(accounts)
            to_acc = random.choice(payees)
            sim_flag = 0

            if random.random() < 0.02:
                device_id = f"new_{uuid.uuid4().hex[:8]}"
            else:
                device_id = random.choice(device_pool)

            r = random.random()
            if r < 0.7:
                amount = random.randint(10, 5000)
                transaction_type = random.choice(["PAYMENT","TRANSFER"])
            elif r < 0.95:
                amount = random.randint(5000, 60000)
                transaction_type = random.choice(["TRANSFER","COLLECT","PAYMENT"])
            else:
                amount = random.randint(60000, 450000)
                transaction_type = random.choice(["TRANSFER","COLLECT"])

            auth_code = random.choices(auth_codes, weights=[0.95,0.05])[0]

        merchant = random.choice(merchant_names)
        category = random.choice(categories)
        ip = random.choice(ip_blocks) + f"{random.randint(0,255)}.{random.randint(0,255)}"
        lat, lon = _rand_geo()
        account_age_days = random.randint(1, 4000)
        credit_limit = random.choice([0, 50000, 100000, 500000, 1000000])
        remote_tool = 1 if transaction_type == "COLLECT" and random.random() < 0.02 else 0
        utilization_rate = round(random.random(), 3) if credit_limit else 0.0

        records.append({
            "txn_id": f"TXN{2000000 + i}",
            "timestamp": ts.isoformat(),
            "from_account_id": from_acc,
            "to_account_id": to_acc,
            "amount": amount,
            "transaction_type": transaction_type,
            "merchant_name": merchant,
            "category": category,
            "ip_address": ip,
            "device_id": device_id,
            "user_lat": lat,
            "user_long": lon,
            "account_age_days": account_age_days,
            "credit_limit": credit_limit,
            "sim_swap_flag": sim_flag,
            "remote_tool_detected": remote_tool,
            "auth_response_code": auth_code,
            "velocity_in_last_hour": 0,
            "utilization_rate": utilization_rate,
            "fraud_label": 1 if is_injected else 0
        })

    df = pd.DataFrame(records)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp_dt").reset_index(drop=True)

    acct_windows = defaultdict(deque)
    df["velocity_in_last_hour"] = 0
    for idx, row in df.iterrows():
        acct = row["from_account_id"]
        ts = row["timestamp_dt"]
        window = acct_windows[acct]

        while window and (ts - window[0]).total_seconds() > 3600:
            window.popleft()

        df.at[idx, "velocity_in_last_hour"] = len(window)
        window.append(ts)

    df["fraud_detected"] = 0

    df.loc[
        (df.sim_swap_flag == 1) &
        (df.device_id.str.contains("new_")) &
        (df.amount > 40000),
        "fraud_detected"
    ] = 1

    cols_order = [
        "txn_id","timestamp","from_account_id","to_account_id","amount",
        "transaction_type","merchant_name","category","ip_address","device_id",
        "user_lat","user_long","account_age_days","credit_limit",
        "sim_swap_flag","remote_tool_detected","auth_response_code",
        "velocity_in_last_hour","utilization_rate",
        "fraud_label","fraud_detected"
    ]

    df_final = df[cols_order].copy()

    # ✅ ALWAYS SAVE CSV
    csv_path = "sim_swap_demo_500.csv"
    df_final.to_csv(csv_path, index=False)

    print(f"\nCSV file saved successfully → {csv_path}\n")
    return df_final


# Run & save immediately
generate_and_save_transactions()
