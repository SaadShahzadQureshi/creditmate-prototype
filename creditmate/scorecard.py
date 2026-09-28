"""Turn a default probability into a credit score and a financing offer."""

import numpy as np

# Scorecard scaling: a score of 600 means 20 farmers repay for every one who
# defaults, and every 40 points doubles those odds.
BASE_SCORE = 600
BASE_ODDS = 20
POINTS_TO_DOUBLE_ODDS = 40

# Offer sizing: finance up to 35% of expected seasonal income, capped.
MAX_SHARE_OF_INCOME = 0.35
FINANCING_CAP_PKR = 600_000


def to_score(pd_default):
    """Map probability of default to a score between 300 and 850 (higher is safer)."""
    p = np.clip(np.asarray(pd_default, dtype=float), 1e-6, 1 - 1e-6)
    odds_repay = (1 - p) / p
    factor = POINTS_TO_DOUBLE_ODDS / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return np.clip(np.round(offset + factor * np.log(odds_repay)), 300, 850).astype(int)


def financing_offer(farmer, pd_default: float, pd_threshold: float) -> dict:
    """Decide, size and structure an offer for one farmer.

    Eligibility comes from risk; the amount comes from capacity. Structures
    are Shariah-compliant sales and leases rather than interest-bearing loans:
    Murabaha (the lender buys inputs and sells them to the farmer at an agreed
    markup, repaid after harvest) or Ijarah (the lender leases a solar pump).
    Illustrative only: a real product needs a Shariah board's approval.
    """
    if pd_default > pd_threshold:
        return {"decision": "decline", "structure": "", "limit_pkr": 0}
    limit = min(MAX_SHARE_OF_INCOME * farmer["seasonal_income_pkr"], FINANCING_CAP_PKR)
    if farmer["irrigation"] in ("diesel_tubewell", "rainfed"):
        structure = "Ijarah: solar pump lease"
    else:
        structure = "Murabaha: seed and fertilizer"
    return {"decision": "approve", "structure": structure, "limit_pkr": int(round(limit, -3))}
