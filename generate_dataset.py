"""
generate_dataset.py
-------------------
Generates a synthetic vehicle rental pricing dataset with 6,000+ records
simulating real-world patterns: seasonality, day-of-week effects,
weather impact, competitor pricing, and demand spikes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

# ── Config ─────────────────────────────────────────────────────────────────
N = 6500
START_DATE = datetime(2022, 1, 1)

VEHICLE_CATEGORIES = {
    "Economy":    {"base": 35,  "demand_factor": 1.10},
    "Compact":    {"base": 45,  "demand_factor": 1.05},
    "Midsize":    {"base": 60,  "demand_factor": 1.00},
    "SUV":        {"base": 90,  "demand_factor": 0.95},
    "Luxury":     {"base": 140, "demand_factor": 0.85},
    "Minivan":    {"base": 75,  "demand_factor": 0.90},
    "Pickup":     {"base": 80,  "demand_factor": 0.92},
}

WEATHER_CONDITIONS = {
    "Sunny":   {"price_adj": 1.08, "prob": 0.35},
    "Cloudy":  {"price_adj": 1.00, "prob": 0.25},
    "Rainy":   {"price_adj": 0.92, "prob": 0.20},
    "Stormy":  {"price_adj": 0.80, "prob": 0.08},
    "Snowy":   {"price_adj": 1.15, "prob": 0.07},
    "Foggy":   {"price_adj": 0.88, "prob": 0.05},
}

KARACHI_HOLIDAYS = {
    (1, 1), (3, 23), (5, 1), (8, 14), (9, 6), (11, 9), (12, 25),
    # Eid al-Fitr approx (varies by year — using fixed offsets)
    (4, 21), (4, 22), (4, 23),
    (6, 28), (6, 29), (6, 30),
    (7, 9),  (7, 10), (7, 11),
}

def is_holiday(dt):
    return (dt.month, dt.day) in KARACHI_HOLIDAYS

def seasonal_factor(dt):
    """Summer + winter peaks."""
    m = dt.month
    if m in [6, 7, 8]:    return 1.20  # Summer road trips
    if m in [12, 1]:      return 1.15  # Winter holidays
    if m in [3, 4]:       return 1.10  # Spring break
    return 1.00

def day_of_week_factor(dow):
    """0=Mon … 6=Sun"""
    return {0: 1.00, 1: 1.02, 2: 1.00, 3: 1.05, 4: 1.18, 5: 1.25, 6: 1.20}[dow]

# ── Generate Records ────────────────────────────────────────────────────────
records = []
for _ in range(N):
    dt = START_DATE + timedelta(days=random.randint(0, 1094))  # 3 years

    vehicle     = random.choice(list(VEHICLE_CATEGORIES.keys()))
    vcat        = VEHICLE_CATEGORIES[vehicle]

    weathers    = list(WEATHER_CONDITIONS.keys())
    w_probs     = [WEATHER_CONDITIONS[w]["prob"] for w in weathers]
    weather     = random.choices(weathers, weights=w_probs)[0]
    wcat        = WEATHER_CONDITIONS[weather]

    holiday     = is_holiday(dt)
    season_f    = seasonal_factor(dt)
    dow_f       = day_of_week_factor(dt.weekday())
    weather_f   = wcat["price_adj"]
    demand_f    = vcat["demand_factor"]
    holiday_f   = 1.22 if holiday else 1.00

    # Competitor base price (market simulation)
    competitor_price = vcat["base"] * (0.88 + np.random.uniform(-0.05, 0.12))

    # Utilization rate 0-100%
    utilization = min(100, max(0,
        60 * season_f * dow_f * holiday_f + np.random.normal(0, 8)
    ))

    # Final price with noise
    price = (
        vcat["base"]
        * season_f
        * dow_f
        * weather_f
        * demand_f
        * holiday_f
        + np.random.normal(0, 3.5)
    )
    price = max(15, round(price, 2))

    records.append({
        "date":                dt.strftime("%Y-%m-%d"),
        "vehicle_category":    vehicle,
        "day_of_week":         dt.strftime("%A"),
        "month":               dt.month,
        "is_weekend":          int(dt.weekday() >= 4),
        "is_holiday":          int(holiday),
        "weather":             weather,
        "season":              ["Winter","Winter","Spring","Spring","Spring","Summer","Summer","Summer","Fall","Fall","Fall","Winter"][dt.month-1],
        "competitor_base_price": round(competitor_price, 2),
        "utilization_rate":    round(utilization, 1),
        "rental_duration_days": random.choices([1,2,3,4,5,6,7,14,30], weights=[20,18,15,12,10,8,7,6,4])[0],
        "daily_rental_price":  price,
    })

df = pd.DataFrame(records)
df.to_csv("rental_pricing_dataset.csv", index=False)

print(f"✓  Dataset saved: rental_pricing_dataset.csv")
print(f"   Rows    : {len(df):,}")
print(f"   Columns : {len(df.columns)}")
print(f"\nPrice stats:")
print(df["daily_rental_price"].describe().round(2))
print(f"\nVehicle distribution:")
print(df["vehicle_category"].value_counts())
