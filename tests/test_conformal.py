"""Unit tests for the conformal core and the guard's decision logic (no models, no data files).

    python -m pytest -q tests
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate import answer_table, conformal_tau  # noqa: E402
from guard import combine, decide  # noqa: E402


def simulate(rng, n_answers: int, informative: bool = True) -> pd.DataFrame:
    """Answers with 3-8 sentences; ~8% of sentences false; a score that is higher for true sentences."""
    rows = []
    for a in range(n_answers):
        for i in range(rng.integers(3, 9)):
            false = rng.random() < 0.08
            s = rng.normal(0.3 if false else 0.7, 0.2) if informative else rng.random()
            rows.append({"id": f"a{a}", "idx": i, "is_false": false, "score": s, "group": "g"})
    return pd.DataFrame(rows)


def coverage_over_draws(alpha: float, informative: bool, draws: int = 400, n_cal: int = 300, n_test: int = 300):
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(draws):
        cal = simulate(rng, n_cal, informative)
        test = simulate(rng, n_test, informative)
        tau = conformal_tau(answer_table(cal, "score")["r"].to_numpy(), alpha)
        covs.append(float((answer_table(test, "score")["r"] <= tau).mean()))
    return float(np.mean(covs)), math.ceil((n_cal + 1) * (1 - alpha)) / (n_cal + 1)


def test_split_conformal_coverage_matches_theory():
    for alpha in (0.05, 0.10, 0.20):
        mean_cov, theory = coverage_over_draws(alpha, informative=True)
        assert mean_cov >= 1 - alpha - 0.01, (alpha, mean_cov)
        assert abs(mean_cov - theory) < 0.01, (alpha, mean_cov, theory)


def test_validity_does_not_depend_on_the_score():
    mean_cov, _ = coverage_over_draws(0.10, informative=False, draws=200)
    assert mean_cov >= 0.89


def test_tau_is_infinite_when_too_few_calibration_answers():
    assert conformal_tau(np.array([0.1, 0.2, 0.3]), 0.05) == math.inf     # k = ceil(4 * 0.95) = 4 > 3


def test_combine_matches_sklearn_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(1)
    feats = ["agree_mean", "agree_frac", "evidence_support", "retrieval_sim", "log_words", "position"]
    X = rng.normal(size=(500, len(feats)))
    y = (X[:, 0] + 0.5 * X[:, 5] + rng.normal(size=500) > 0).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)
    sc, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    th = {"features": feats, "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
          "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    ours = combine({f: X[:, j] for j, f in enumerate(feats)}, th)
    assert np.allclose(ours, model.predict_proba(X)[:, 1], atol=1e-10)


def test_decide_ship_review_block():
    th = {"tau": {0.05: 0.6, 0.10: 0.4}, "ship_lambda": 0.8, "main_alpha": 0.05}
    assert decide(np.array([0.9, 0.85, 0.95]), th)["decision"] == "ship"
    r = decide(np.array([0.9, 0.5, 0.7]), th)
    assert r["decision"] == "review" and r["kept"].tolist() == [True, False, True]
    assert decide(np.array([0.3, 0.5]), th)["decision"] == "block"
    # a looser alpha keeps more sentences, but ship/review/block stays at the calibrated main alpha
    r = decide(np.array([0.3, 0.5]), th, alpha=0.10)
    assert r["kept"].tolist() == [False, True] and r["decision"] == "block"
