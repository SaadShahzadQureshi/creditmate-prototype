"""Simulate a population of smallholder farmers in Sindh, Pakistan.

Every number in this module is simulated. The prototype has no real farm or
repayment data yet, so it builds a stand-in population whose relationships
reflect CreditMate's core assumption: a farmer's ability to repay depends on
what the farm produces and how steady that income is, far more than on how much
land they hold or whether they hold a title to it.

Swap this module for real data (lender repayment records joined to crop,
irrigation and livestock data) and the rest of the pipeline stays the same.
"""

import numpy as np
import pandas as pd

CROPS = ["wheat", "cotton", "rice", "sugarcane", "mango", "banana"]
CROP_SHARES = [0.34, 0.20, 0.15, 0.11, 0.12, 0.08]

# Illustrative gross revenue per acre per season, in PKR.
REVENUE_PER_ACRE = {
    "wheat": 110_000,
    "cotton": 140_000,
    "rice": 120_000,
    "sugarcane": 200_000,
    "mango": 180_000,
    "banana": 220_000,
}

IRRIGATION = ["canal", "diesel_tubewell", "solar_pump", "rainfed"]
IRRIGATION_SHARES = [0.45, 0.28, 0.10, 0.17]

# How dependable each water source is across a season (0 to 1).
IRRIGATION_RELIABILITY = {
    "canal": 0.65,
    "diesel_tubewell": 0.75,
    "solar_pump": 0.90,
    "rainfed": 0.35,
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate_farmers(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Return one row per simulated farmer, including whether they defaulted.

    Columns a lender could observe: land_acres, has_land_title, crop,
    irrigation, yield_index, yield_volatility, herd_size, herd_condition,
    wallet_months, mobile_inflow_pkr, sells_through_arthi, seasonal_income_pkr.
    Outcome: default_12m (fell 90+ days behind within 12 months).
    """
    rng = np.random.default_rng(seed)

    # Land held, in acres. Most smallholders farm under 12.5 acres.
    land_acres = np.round(np.clip(rng.lognormal(mean=1.4, sigma=0.65, size=n), 0.5, 40), 1)

    # Larger farms are more likely to hold a clear title. Many small farmers
    # lease, share-crop or farm family land that isn't in their name.
    has_land_title = rng.random(n) < _sigmoid((land_acres - 7) / 2.5)

    crop = rng.choice(CROPS, size=n, p=CROP_SHARES)
    irrigation = rng.choice(IRRIGATION, size=n, p=IRRIGATION_SHARES)
    reliability = np.array([IRRIGATION_RELIABILITY[i] for i in irrigation]) + rng.normal(0, 0.05, n)

    # Farming practice: a hidden trait no lender sees directly. It shows up
    # only through what the farm produces.
    practice = rng.normal(0, 1, n)

    # Yield over the last three seasons against the district average (1.0).
    yield_index = np.clip(
        1 + 0.15 * practice + 0.35 * (reliability - 0.65) + rng.normal(0, 0.08, n), 0.4, 1.7
    )

    # How much yields swung across those seasons (coefficient of variation).
    yield_volatility = np.clip(
        0.42 - 0.30 * reliability - 0.04 * practice + rng.normal(0, 0.05, n), 0.04, 0.6
    )

    # Livestock: buffalo and cattle, with a body condition score from 1 (poor)
    # to 5 (excellent). A later version would score this from a phone photo.
    herd_size = rng.poisson(lam=np.clip(1.5 + 0.25 * land_acres, 0, 12))
    herd_condition = np.clip(np.round((3 + 0.5 * practice + rng.normal(0, 0.6, n)) * 2) / 2, 1, 5)
    herd_condition = np.where(herd_size == 0, np.nan, herd_condition)

    # Net seasonal farm income, in PKR: crops plus livestock.
    revenue_per_acre = np.array([REVENUE_PER_ACRE[c] for c in crop])
    livestock_income = herd_size * 25_000 * np.nan_to_num(herd_condition, nan=0) / 3
    seasonal_income = land_acres * revenue_per_acre * yield_index * 0.45 + livestock_income

    # Mobile wallet use: months of history and average monthly inflows.
    wallet_months = rng.integers(0, 37, n)
    share_through_wallet = np.clip(rng.normal(0.5, 0.2, n), 0.05, 1.2)
    mobile_inflow = np.where(wallet_months > 0, seasonal_income / 6 * share_through_wallet, 0.0)

    # Selling through an arthi, a commission agent who advanced cash against the crop.
    sells_through_arthi = rng.random(n) < np.clip(0.55 - 0.15 * practice, 0.1, 0.9)

    # The outcome. Risk is driven by yield level and stability, herd health,
    # wallet history and dependence on an arthi. Land size has only a small effect.
    herd_term = np.nan_to_num(herd_condition, nan=3.0) - 3
    logit = (
        -1.75
        - 3.0 * (yield_index - 1)
        + 5.0 * (yield_volatility - 0.2)
        - 0.35 * herd_term
        - 0.012 * np.minimum(wallet_months, 24)
        + 0.45 * sells_through_arthi
        - 0.10 * np.log(land_acres)
        + rng.normal(0, 0.35, n)
    )
    default_12m = rng.random(n) < _sigmoid(logit)

    return pd.DataFrame(
        {
            "farmer_id": [f"F{i:05d}" for i in range(1, n + 1)],
            "land_acres": land_acres,
            "has_land_title": has_land_title,
            "crop": crop,
            "irrigation": irrigation,
            "yield_index": np.round(yield_index, 3),
            "yield_volatility": np.round(yield_volatility, 3),
            "herd_size": herd_size,
            "herd_condition": herd_condition,
            "wallet_months": wallet_months,
            "mobile_inflow_pkr": np.round(mobile_inflow, -2),
            "sells_through_arthi": sells_through_arthi,
            "seasonal_income_pkr": np.round(seasonal_income, -3),
            "default_12m": default_12m,
        }
    )
