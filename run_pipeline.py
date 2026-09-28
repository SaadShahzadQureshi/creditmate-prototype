"""Run CreditMate end to end: simulate farmers, train, compare, score, and save results.

Usage:
    python run_pipeline.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from creditmate import charts, model, scorecard, simulate

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

TARGET_DEFAULT_RATE = 0.10  # the lender's appetite: at most 1 in 10 approved farmers defaults


def main():
    DATA_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    # 1. Simulate 5,000 farmers and save them so anyone can inspect the data.
    farmers = simulate.simulate_farmers(n=5000, seed=42)
    farmers.to_csv(DATA_DIR / "farmers_simulated.csv", index=False)

    # 2. Split farmers 60/20/20: train the model, pick the cutoff, then test.
    train, rest = train_test_split(farmers, test_size=0.4, stratify=farmers["default_12m"], random_state=7)
    valid, test = train_test_split(rest, test_size=0.5, stratify=rest["default_12m"], random_state=7)

    # 3. Train on the training farmers only.
    fitted = model.train_model(model.build_features(train), train["default_12m"].astype(int))

    # 4. Choose the cutoff on the validation farmers.
    pd_valid = fitted.predict_proba(model.build_features(valid))[:, 1]
    cutoff = model.cutoff_for_default_rate(pd_valid, valid["default_12m"].to_numpy(), TARGET_DEFAULT_RATE)

    # 5. Judge everything on the test farmers, which the model has never seen.
    y_test = test["default_12m"].to_numpy().astype(int)
    pd_test = fitted.predict_proba(model.build_features(test))[:, 1]
    results = model.compare_with_collateral(test, pd_test, y_test, cutoff)
    results["target_default_rate"] = TARGET_DEFAULT_RATE

    # 6. Score every test farmer and make an offer.
    scores = scorecard.to_score(pd_test)
    offers = [scorecard.financing_offer(row, p, cutoff) for (_, row), p in zip(test.iterrows(), pd_test)]
    decisions = test[["farmer_id", "land_acres", "has_land_title", "crop", "irrigation",
                      "seasonal_income_pkr"]].reset_index(drop=True)
    decisions["probability_of_default"] = pd_test.round(3)
    decisions["score"] = scores
    decisions = pd.concat([decisions, pd.DataFrame(offers)], axis=1)
    decisions.head(25).to_csv(OUT_DIR / "sample_decisions.csv", index=False)

    # 7. Save metrics and charts.
    drivers = model.risk_drivers(fitted, model.FEATURES)
    results["risk_drivers"] = drivers.round(3).to_dict()
    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    charts.roc_chart(y_test, pd_test, model.collateral_rule(test), results["auc_farm_data_model"],
                     OUT_DIR / "roc_curve.png")
    charts.comparison_chart(results, OUT_DIR / "comparison.png")
    charts.drivers_chart(drivers, OUT_DIR / "risk_drivers.png")
    charts.score_chart(scores, y_test, OUT_DIR / "score_distribution.png")

    # 8. Print a short summary.
    rule = results["land_title_rule"]
    same_rate = results["model_at_same_approval_rate"]
    target = results["model_at_target_default_rate"]
    print(f"Farm-data model AUC on test farmers: {results['auc_farm_data_model']:.2f}")
    print(f"Land-title rule: approves {rule['approval_rate']:.0%}; "
          f"{rule['default_rate_approved']:.0%} of them default; "
          f"{rule['share_approved_without_title']:.0%} have no title")
    print(f"Model, same approval rate: {same_rate['default_rate_approved']:.0%} default; "
          f"{same_rate['share_approved_without_title']:.0%} have no title")
    print(f"Model, {TARGET_DEFAULT_RATE:.0%} target default rate: approves {target['approval_rate']:.0%}; "
          f"{target['default_rate_approved']:.0%} default; "
          f"{target['share_approved_without_title']:.0%} have no title")
    print(f"Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
