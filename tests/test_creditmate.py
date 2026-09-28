"""Checks that the pipeline behaves as described in the README."""

import numpy as np
from sklearn.model_selection import train_test_split

from creditmate import model, scorecard, simulate


def test_simulation_is_reproducible():
    a = simulate.simulate_farmers(n=800, seed=1)
    b = simulate.simulate_farmers(n=800, seed=1)
    assert a.equals(b)
    assert len(a) == 800
    assert 0.10 < a["default_12m"].mean() < 0.30


def test_model_never_sees_land():
    X = model.build_features(simulate.simulate_farmers(n=300, seed=2))
    assert not any("land" in column for column in X.columns)
    assert not X.isna().any().any()


def test_model_beats_the_land_title_rule():
    farmers = simulate.simulate_farmers(n=4000, seed=3)
    train, test = train_test_split(farmers, test_size=0.3, stratify=farmers["default_12m"], random_state=0)
    fitted = model.train_model(model.build_features(train), train["default_12m"].astype(int))
    pd_test = fitted.predict_proba(model.build_features(test))[:, 1]
    y = test["default_12m"].to_numpy().astype(int)
    results = model.compare_with_collateral(test, pd_test, y, target_cutoff=0.1)
    assert results["auc_farm_data_model"] > 0.7
    assert (results["model_at_same_approval_rate"]["default_rate_approved"]
            < results["land_title_rule"]["default_rate_approved"])


def test_scores_fall_as_risk_rises():
    scores = scorecard.to_score([0.02, 0.10, 0.30, 0.60])
    assert list(scores) == sorted(scores, reverse=True)
    assert scorecard.to_score([1 / 21])[0] == scorecard.BASE_SCORE


def test_declined_farmers_get_no_offer():
    farmer = {"seasonal_income_pkr": 400_000, "irrigation": "canal"}
    assert scorecard.financing_offer(farmer, 0.30, 0.10)["limit_pkr"] == 0
    offer = scorecard.financing_offer(farmer, 0.05, 0.10)
    assert offer["decision"] == "approve"
    assert offer["limit_pkr"] == 140_000
    assert offer["structure"].startswith("Murabaha")
