1. Multiple international transactions on same day

Description:
5 transactions occur within 24 hours:

3 in home country

2 in different countries

All within short time intervals.

Condition JSON example:

{
  "pattern": "international_mix",
  "total_tx": 5,
  "same_country": 3,
  "foreign_countries": 2,
  "time_window_hours": 24
}

2. Exceeds transaction limit within minutes

Description:
Customer makes multiple high-value transactions in same minute or same hour.

Condition:

{
  "pattern": "rapid_high_value",
  "max_amount": 5000,
  "tx_count": 4,
  "time_window_min": 10
}

3. Multiple declined attempts then success

Description:
Fraudsters often test cards.

Condition:

{
  "pattern": "decline_then_success",
  "declined_attempts": 3,
  "final_success": true,
  "time_window_min": 5
}

4. Sudden spending spike

If user usually spends ≤ ₹2000/day but suddenly does ₹50,000 in 1 hour.

Condition:

{
  "pattern": "spending_spike",
  "baseline_daily_avg": 2000,
  "sudden_spend": 50000,
  "time_window_hours": 1
}

5. Multiple transactions from different locations within hours

Example:

10:00 AM – Hyderabad

10:15 AM – Mumbai

10:18 AM – Dubai

Impossible travel → fraud.

Condition:

{
  "pattern": "impossible_travel",
  "locations": ["Hyderabad", "Mumbai", "Dubai"],
  "time_window_min": 30
}

6. Card used at different merchant types rapidly

Fraudster tests card with small then big purchases.

Condition:

{
  "pattern": "merchant_mix_fast",
  "merchant_types": ["grocery", "fuel", "electronics", "gaming"],
  "tx_count": 4,
  "time_window_min": 20
}

7. High-risk merchant + unusual hour

E.g., gambling → 3 AM.

Condition:

{
  "pattern": "night_risky_merchant",
  "merchant_type": "gambling",
  "time_range": "00:00-05:00"
}

8. Very low device trust score

Fraud often comes from unknown devices.

Condition:

{
  "pattern": "low_device_trust",
  "device_trust_score": "< 0.2"
}

9. Suspicious refund behavior

Refund more than original purchase.

Condition:

{
  "pattern": "fraud_refund",
  "purchase_amount": 1500,
  "refund_amount": 3000
}

10. Many micro-transactions (fraud testing)

₹1, ₹2, ₹5 transactions to test validity.

Condition:

{
  "pattern": "micro_test",
  "amounts": [1, 2, 5],
  "tx_count": 5,
  "time_window_min": 15
}
