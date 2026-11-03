"""Calibrate the conformal claim filter and produce every result, table and figure in results/.

    python calibrate.py                         # false = contradicted by the doctor-written references
    python calibrate.py --alpha 0.05            # main error level (the one the demo and README use)
    python calibrate.py --false-label strict    # false = anything not supported (comparison only)
    python calibrate.py --labels v2 --alpha 0.10   # sensitivity: original broad v2 labels -> results/v2_labels/

Labels are read from data/claims.parquet (so a rule applied with audit_labels.py takes effect without
re-running score_claims.py); scores from data/scores.parquet.

Splits (frozen in data/questions_final.jsonl):
  train      -> fit the score combiner (logistic regression)          never used for thresholds
  calibrate  -> conformal thresholds tau(alpha)                       never used for fitting
  test       -> every reported number                                 never used for either

Combiner features are ONLY: agreement with resamples, MedlinePlus evidence support, retrieval
similarity, sentence length and position. The reference-based NLI scores are excluded because the
labels come from them (using them would leak the answer).

Results: results/summary.md, results/summary.json, results/thresholds.json,
results/test_claims.parquet and the figures in results/figures/.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from cg_common import ROOT

ALPHAS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]   # replaced in main() by alpha_grid(main alpha)


def alpha_grid(main: float) -> list[float]:
    """Error levels around the main one: where the filter actually has work to do."""
    return sorted({round(main * m, 4) for m in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0)})
MAIN_ALPHA = 0.10
DELTA = 0.05                    # Learn-Then-Test confidence level
N_RESPLITS = 1000
SEED = 20261102
FEATURES = ["agree_mean", "agree_frac", "evidence_support", "retrieval_sim", "log_words", "position"]


# ----------------------------------------------------------------------------- data
def load(false_label: str, labels: str = "applied") -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "scores.parquet")
    if labels == "v2":
        lab = pd.read_parquet(ROOT / "data" / "claims_v2.parquet").assign(label_rule="v2")
    else:
        lab = pd.read_parquet(ROOT / "data" / "claims.parquet")
    lab_cols = ["id", "idx", "label"] + (["label_rule"] if "label_rule" in lab.columns else [])
    df = df.drop(columns=[c for c in ("label", "label_rule") if c in df.columns]).merge(
        lab[lab_cols], on=["id", "idx"], how="inner")
    if "label_rule" not in df.columns:
        df["label_rule"] = "v2"
        rule_file = ROOT / "data" / "label_rule.json"
        if rule_file.exists():
            rule = json.loads(rule_file.read_text(encoding="utf-8"))["rule"]
            raise SystemExit(f"data/claims.parquet has fresh v2 labels but you chose rule '{rule}' earlier: "
                             f"run `python audit_labels.py --apply {rule}` first")
    df["is_false"] = (df["label"] == "contradicted") if false_label == "contradicted" else (df["label"] != "supported")
    df["log_words"] = np.log(df["n_words"].clip(lower=1))
    n_in_answer = df.groupby("id")["idx"].transform("count")
    df["position"] = df["idx"] / n_in_answer.clip(lower=1)
    return df


def fit_combiner(train: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model.fit(train[FEATURES].to_numpy(), (~train["is_false"]).astype(int).to_numpy())
    return model


# ----------------------------------------------------------------------------- conformal core
def answer_table(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """One row per answer: r = highest score among its false claims (-inf if none)."""
    g = df.assign(_fs=np.where(df["is_false"], df[score], -np.inf)).groupby("id")
    return pd.DataFrame({"r": g["_fs"].max(), "group": g["group"].first(), "n_claims": g.size()})


def conformal_tau(r_cal: np.ndarray, alpha: float) -> float:
    n = len(r_cal)
    k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else float(np.sort(r_cal)[k - 1])


def evaluate(df: pd.DataFrame, score: str, tau: float) -> dict:
    kept = df[score] > tau
    ans = answer_table(df, score)
    clean = ans["r"] <= tau
    kept_per_answer = kept.groupby(df["id"]).sum()
    true = ~df["is_false"]
    return {"coverage": float(clean.mean()), "n_answers": int(len(ans)), "n_clean": int(clean.sum()),
            "claim_retention": float(kept.mean()),
            "true_claim_retention": float((kept & true).sum() / max(1, true.sum())),
            "nonempty_answers": float((kept_per_answer > 0).mean())}


def clopper_pearson(x: int, n: int, level: float = 0.95) -> tuple[float, float]:
    from scipy.stats import beta
    a = (1 - level) / 2
    lo = 0.0 if x == 0 else float(beta.ppf(a, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - a, x + 1, n - x))
    return lo, hi


def ltt_threshold(df_cal: pd.DataFrame, score: str, alpha: float, delta: float) -> float:
    """Learn-Then-Test with binomial p-values and fixed-sequence testing (strictest threshold first)."""
    from scipy.stats import binom
    r = answer_table(df_cal, score)["r"].to_numpy()
    n = len(r)
    grid = np.unique(np.concatenate([[np.inf], df_cal[score].to_numpy()]))[::-1]   # strict -> lenient
    chosen = math.inf
    for lam in grid:
        risk_count = int((r > lam).sum())          # answers that would still contain a false claim
        p = binom.cdf(risk_count, n, alpha)        # H0: true risk > alpha
        if p <= delta:
            chosen = float(lam)
        else:
            break
    return chosen


def crc_threshold(losses_by_lambda, grid, n: int, alpha: float, B: float = 1.0) -> float:
    """Conformal risk control: smallest lambda with (n/(n+1)) * mean loss + B/(n+1) <= alpha."""
    best = math.inf
    for lam, risk in zip(grid, losses_by_lambda):     # grid ascending, loss non-increasing in lambda
        if (n / (n + 1)) * risk + B / (n + 1) <= alpha:
            best = float(lam)
            break
    return best


# ----------------------------------------------------------------------------- main
def main() -> None:
    from scipy.stats import beta as beta_dist
    from scipy.stats import betabinom
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--false-label", choices=["contradicted", "strict"], default="contradicted")
    ap.add_argument("--alpha", type=float, default=0.10, help="main error level for the detailed results")
    ap.add_argument("--labels", choices=["applied", "v2"], default="applied",
                    help="'v2' = sensitivity run on the original broad labels (written to results/v2_labels)")
    args = ap.parse_args()
    global MAIN_ALPHA, ALPHAS
    MAIN_ALPHA = args.alpha
    ALPHAS = alpha_grid(MAIN_ALPHA)
    out = ROOT / "results" if args.false_label == "contradicted" else ROOT / "results" / "strict"
    if args.labels == "v2":
        out = ROOT / "results" / "v2_labels"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df = load(args.false_label, args.labels)
    train, cal, test = (df[df["split"] == s].copy() for s in ("train", "calibrate", "test"))
    model = fit_combiner(train)
    for part in (df, train, cal, test):
        part["combined"] = model.predict_proba(part[FEATURES].to_numpy())[:, 1]
        part["agreement"] = part["agree_mean"]
        part["evidence"] = part["evidence_support"]
    rand = {k: v for k, v in zip(df.index, rng.uniform(size=len(df)))}
    for part in (df, train, cal, test):
        part["random"] = part.index.map(rand)

    S = {"false_label": args.false_label, "label_rule": str(df["label_rule"].iloc[0]), "main_alpha": MAIN_ALPHA,
         "counts": {}, "raw": {}, "conformal": {}, "naive": {},
         "ablation": {}, "random_control": {}, "per_group": {}, "ltt": {}, "crc_lenient": {},
         "abstention": {}, "resplits": {}}
    for name, part in (("train", train), ("calibrate", cal), ("test", test)):
        S["counts"][name] = {"answers": int(part["id"].nunique()), "claims": int(len(part)),
                             "false_claims": int(part["is_false"].sum())}
    raw_err = float(answer_table(test, "combined")["r"].gt(-np.inf).mean())
    S["raw"] = {"test_answers_with_false_claim": raw_err, "test_false_claim_share": float(test["is_false"].mean())}

    # 1. conformal thresholds and test evaluation
    r_cal = answer_table(cal, "combined")["r"].to_numpy()
    taus = {}
    for a in ALPHAS:
        tau = conformal_tau(r_cal, a)
        taus[a] = tau
        ev = evaluate(test, "combined", tau)
        lo, hi = clopper_pearson(ev["n_clean"], ev["n_answers"])
        S["conformal"][str(a)] = {"tau": tau, **ev, "ci95": [lo, hi], "target": 1 - a,
                                  "filter_needed": bool(tau > -math.inf)}

    # 2. naive fixed threshold and keep-everything
    S["naive"] = {"fixed_0.5": evaluate(test, "combined", 0.5), "keep_all": evaluate(test, "combined", -math.inf)}

    # 3. ablation: AUROC on test claims + retention at the main alpha, each score calibrated separately
    for sc in ("combined", "agreement", "evidence", "random"):
        auroc = float(roc_auc_score((~test["is_false"]).astype(int), test[sc])) if test["is_false"].any() else None
        tau = conformal_tau(answer_table(cal, sc)["r"].to_numpy(), MAIN_ALPHA)
        ev = evaluate(test, sc, tau)
        S["ablation"][sc] = {"auroc": auroc, "tau": tau, **ev}
    S["random_control"] = S["ablation"]["random"]

    # 4. per-group: marginal threshold vs group-wise (Mondrian) threshold
    for gname in sorted(df["group"].unique()):
        cg, tg = cal[cal["group"] == gname], test[test["group"] == gname]
        tau_g = conformal_tau(answer_table(cg, "combined")["r"].to_numpy(), MAIN_ALPHA)
        S["per_group"][gname] = {"marginal": evaluate(tg, "combined", taus[MAIN_ALPHA]),
                                 "mondrian": {"tau": tau_g, **evaluate(tg, "combined", tau_g)},
                                 "n_cal": int(cg["id"].nunique()), "n_test": int(tg["id"].nunique())}

    # 5. Learn-Then-Test (PAC: with prob >= 1-delta over calibration, risk <= alpha)
    lam = ltt_threshold(cal, "combined", MAIN_ALPHA, DELTA)
    S["ltt"] = {"alpha": MAIN_ALPHA, "delta": DELTA, "lambda": lam, **evaluate(test, "combined", lam)}

    # 6. lenient mode: CRC on (false claims kept) / (claims in answer)
    grid = np.unique(np.concatenate([[-np.inf], cal["combined"].to_numpy(), [np.inf]]))
    per_ans = cal.groupby("id")
    n_cal = cal["id"].nunique()
    risks = []
    for lamb in grid:
        lost = ((cal["combined"] > lamb) & cal["is_false"]).groupby(cal["id"]).sum() / per_ans.size()
        risks.append(float(lost.mean()))
    lam_crc = crc_threshold(risks, grid, n_cal, MAIN_ALPHA)
    kept = test["combined"] > lam_crc
    loss_test = (kept & test["is_false"]).groupby(test["id"]).sum() / test.groupby("id").size()
    S["crc_lenient"] = {"alpha": MAIN_ALPHA, "lambda": lam_crc, "test_mean_loss": float(loss_test.mean()),
                        "claim_retention": float(kept.mean())}

    # 7. ship / review / block: answer score = weakest claim; CRC on P(shipped and wrong)
    def answer_scores(part):
        g = part.groupby("id")
        return pd.DataFrame({"a": g["combined"].min(), "wrong": g["is_false"].any()})
    ac, at = answer_scores(cal), answer_scores(test)
    grid_a = np.unique(np.concatenate([ac["a"].to_numpy(), [np.inf]]))
    risks_a = [float(((ac["a"] >= l) & ac["wrong"]).mean()) for l in grid_a]
    lam_ship = crc_threshold(risks_a, grid_a, len(ac), MAIN_ALPHA)
    ship = at["a"] >= lam_ship
    kept_main = (test["combined"] > taus[MAIN_ALPHA]).groupby(test["id"]).sum()
    block = (~ship) & (kept_main.reindex(at.index).fillna(0) == 0)
    review = (~ship) & (~block)
    S["abstention"] = {"alpha": MAIN_ALPHA, "lambda_ship": lam_ship, "share_ship": float(ship.mean()),
                       "share_review": float(review.mean()), "share_block": float(block.mean()),
                       "p_shipped_and_wrong": float((ship & at["wrong"]).mean()),
                       "p_wrong_given_shipped": float((ship & at["wrong"]).sum() / max(1, ship.sum()))}

    # decision and filter matrices on the test set (what the figure matrices.png shows)
    dec = pd.Series(np.where(ship, "ship", np.where(block, "block", "review")), index=at.index)
    kept_claim = test["combined"] > taus[MAIN_ALPHA]
    S["matrices"] = {
        "alpha": MAIN_ALPHA, "n_answers": int(len(at)), "n_sentences": int(len(test)),
        "answers": {d: {"clean": int(((dec == d) & ~at["wrong"]).sum()),
                        "contradicted": int(((dec == d) & at["wrong"]).sum())} for d in ("ship", "review", "block")},
        "sentences": {name: {"fine": int((mask & ~test["is_false"]).sum()), "contradicted": int((mask & test["is_false"]).sum())}
                      for name, mask in (("kept", kept_claim), ("dropped", ~kept_claim))}}

    # 8. 1,000 random re-splits of calibrate+test (combiner fixed from train)
    pool = pd.concat([cal, test])
    pool_ans = answer_table(pool, "combined")
    r_all = pool_ans["r"].to_numpy()
    n_c, n_t = cal["id"].nunique(), test["id"].nunique()
    # retention of each score over the same re-splits: a fairer comparison than the single test split
    scores = ("combined", "agreement", "evidence", "random")
    pos = {a: i for i, a in enumerate(pool_ans.index)}
    claim_ans = pool["id"].map(pos).to_numpy()
    R = {sc: answer_table(pool, sc)["r"].reindex(pool_ans.index).to_numpy() for sc in scores}
    V = {sc: pool[sc].to_numpy() for sc in scores}
    covs, ret, raw = [], {sc: [] for sc in scores}, []
    by_a = {a: {"cov": [], "kept": [], "nonempty": []} for a in ALPHAS}
    for _ in range(N_RESPLITS):
        perm = rng.permutation(len(r_all))
        cal_a, test_a = perm[:n_c], perm[n_c:n_c + n_t]
        tau = conformal_tau(r_all[cal_a], MAIN_ALPHA)
        covs.append(float((r_all[test_a] <= tau).mean()))
        raw.append(float((r_all[test_a] == -np.inf).mean()))
        in_test = np.zeros(len(r_all), bool)
        in_test[test_a] = True
        cm = in_test[claim_ans]
        for sc in scores:
            t_sc = tau if sc == "combined" else conformal_tau(R[sc][cal_a], MAIN_ALPHA)
            ret[sc].append(float((V[sc][cm] > t_sc).mean()))
        for a in ALPHAS:                                   # coverage and cost at every alpha, same split
            t_a = conformal_tau(r_all[cal_a], a)
            kept_c = V["combined"][cm] > t_a
            by_a[a]["cov"].append(float((r_all[test_a] <= t_a).mean()))
            by_a[a]["kept"].append(float(kept_c.mean()))
            per = np.bincount(claim_ans[cm], weights=kept_c, minlength=len(r_all))[test_a]
            by_a[a]["nonempty"].append(float((per > 0).mean()))
    covs = np.array(covs)

    def band(v):
        return {"mean": float(np.mean(v)), "p05": float(np.quantile(v, 0.05)), "p95": float(np.quantile(v, 0.95))}
    k = math.ceil((n_c + 1) * (1 - MAIN_ALPHA))
    S["resplits"] = {"alpha": MAIN_ALPHA, "n": N_RESPLITS, "mean": float(covs.mean()),
                     "p05": float(np.quantile(covs, 0.05)), "p95": float(np.quantile(covs, 0.95)),
                     "share_below_target": float((covs < 1 - MAIN_ALPHA).mean()),
                     "theory_mean": k / (n_c + 1), "theory_beta": [k, n_c + 1 - k],
                     "retention": {sc: band(v) for sc, v in ret.items()}, "raw_mean": float(np.mean(raw)),
                     "by_alpha": [{"alpha": a, "target": 1 - a, "coverage": band(d["cov"]), "kept": band(d["kept"]),
                                   "nonempty": float(np.mean(d["nonempty"]))} for a, d in by_a.items()]}

    # ------------------------------------------------------------------ save
    test_out = test[["id", "group", "idx", "text", "start", "end", "label", "is_false", "agree_mean",
                     "evidence_support", "combined"]].copy()
    for a in ALPHAS:
        test_out[f"kept_a{round(a * 1000):03d}"] = test_out["combined"] > taus[a]
    test_out.to_parquet(out / "test_claims.parquet", index=False)
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    thresholds = {"false_label": args.false_label, "label_rule": S["label_rule"], "main_alpha": MAIN_ALPHA,
                  "features": FEATURES,
                  "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                  "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
                  "tau": {str(a): t for a, t in taus.items()}, "ltt_lambda": lam, "crc_lenient_lambda": lam_crc,
                  "ship_lambda": lam_ship, "n_calibration_answers": int(n_c)}
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=1, default=float))
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    write_markdown(S, out / "summary.md", thresholds)
    make_figures(S, covs, n_c, n_t, k, figs, beta_dist, betabinom)
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"\nresults written to {out}")


# ----------------------------------------------------------------------------- reporting
def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def write_markdown(S: dict, dest, th: dict) -> None:
    c, raw = S["counts"], S["raw"]
    L = [f"# Conformal Guard results (false = {S['false_label']}, label rule {S['label_rule']}, main alpha {S['main_alpha']})", "",
         f"Answers: train {c['train']['answers']}, calibrate {c['calibrate']['answers']}, test {c['test']['answers']}. "
         f"Claims: {c['train']['claims'] + c['calibrate']['claims'] + c['test']['claims']}.", "",
         f"**Raw Muse Spark answers (test): {pct(raw['test_answers_with_false_claim'])} contain at least one false claim** "
         f"({pct(raw['test_false_claim_share'])} of claims are false).", "",
         "## Guarantee on held-out test answers", "",
         "| alpha | target 1-alpha | test coverage | 95% CI | claims kept | true claims kept | non-empty answers |",
         "|---|---|---|---|---|---|---|"]
    for a, r in S["conformal"].items():
        note = "" if r["filter_needed"] else " (no filtering needed)"
        L.append(f"| {a} | {pct(r['target'])} | {pct(r['coverage'])}{note} | {pct(r['ci95'][0])}–{pct(r['ci95'][1])} | "
                 f"{pct(r['claim_retention'])} | {pct(r['true_claim_retention'])} | {pct(r['nonempty_answers'])} |")
    rs = S["resplits"]
    L += ["", f"## {rs['n']} random re-splits (alpha = {rs['alpha']})", "",
          f"Mean test coverage {pct(rs['mean'])}; conformal theory predicts {pct(rs['theory_mean'])}. "
          f"Individual re-splits range {pct(rs['p05'])}–{pct(rs['p95'])} (middle 90%) because each test set has only "
          f"{c['test']['answers']} answers; the guarantee is about the average, and the histogram matches the predicted "
          "Beta-binomial curve.", "",
          "Share of claims kept, averaged over the same re-splits (each score calibrated separately): "
          + ", ".join(f"{sc} {pct(v['mean'])} ({pct(v['p05'])}–{pct(v['p95'])})" for sc, v in rs["retention"].items())
          + ".", "",
          f"Every alpha, averaged over the same {rs['n']} re-splits (unfiltered answers: {pct(rs['raw_mean'])} clean):", "",
          "| alpha | target | mean coverage | middle 90% | sentences kept | answers with a sentence kept |",
          "|---|---|---|---|---|---|"]
    for r in rs["by_alpha"]:
        L.append(f"| {r['alpha']:g} | {pct(r['target'])} | {pct(r['coverage']['mean'])} | "
                 f"{pct(r['coverage']['p05'])}–{pct(r['coverage']['p95'])} | {pct(r['kept']['mean'])} | "
                 f"{pct(r['nonempty'])} |")
    L += ["",
          "## Baselines", "",
          f"- Keep everything: coverage {pct(S['naive']['keep_all']['coverage'])}",
          f"- Fixed cutoff (combined score > 0.5), no calibration: coverage {pct(S['naive']['fixed_0.5']['coverage'])}, "
          f"claims kept {pct(S['naive']['fixed_0.5']['claim_retention'])}",
          f"- Random score + conformal: coverage {pct(S['random_control']['coverage'])} (still valid), "
          f"claims kept {pct(S['random_control']['claim_retention'])} (useless)", "",
          f"## Which scores matter (alpha = {MAIN_ALPHA})", "",
          "| score | AUROC (true vs false claims) | test coverage | claims kept |", "|---|---|---|---|"]
    for sc, r in S["ablation"].items():
        L.append(f"| {sc} | {'–' if r['auroc'] is None else f'{r['auroc']:.3f}'} | {pct(r['coverage'])} | {pct(r['claim_retention'])} |")
    L += ["", f"## Per group (alpha = {MAIN_ALPHA})", "",
          "| group | calibration answers | test answers | coverage (shared threshold) | coverage (group threshold) | claims kept (group threshold) |",
          "|---|---|---|---|---|---|"]
    for g, r in S["per_group"].items():
        L.append(f"| {g} | {r['n_cal']} | {r['n_test']} | {pct(r['marginal']['coverage'])} | "
                 f"{pct(r['mondrian']['coverage'])} | {pct(r['mondrian']['claim_retention'])} |")
    lt, cr, ab = S["ltt"], S["crc_lenient"], S["abstention"]
    L += ["", "## Other guarantee modes", "",
          f"- Learn-Then-Test (with {pct(1 - lt['delta'])} confidence, error <= {lt['alpha']}): test coverage {pct(lt['coverage'])}, "
          f"claims kept {pct(lt['claim_retention'])}",
          f"- Lenient (expected share of false claims per answer <= {cr['alpha']}): test value {pct(cr['test_mean_loss'])}, "
          f"claims kept {pct(cr['claim_retention'])}",
          f"- Ship / review / block: ship {pct(ab['share_ship'])}, review {pct(ab['share_review'])}, block {pct(ab['share_block'])}; "
          f"P(shipped and wrong) = {pct(ab['p_shipped_and_wrong'])} (bound {ab['alpha']}), "
          f"P(wrong | shipped) = {pct(ab['p_wrong_given_shipped'])}", ""]
    M = S["matrices"]
    na, ns = M["n_answers"], M["n_sentences"]
    L += [f"## Decision matrix ({na} test answers, alpha = {M['alpha']})", "",
          "| decision | no contradicted sentence | has a contradicted sentence |", "|---|---|---|"]
    for d, label in (("ship", "shipped unchanged"), ("review", "filtered, sent for review"), ("block", "blocked")):
        r = M["answers"][d]
        L.append(f"| {label} | {r['clean']} ({pct(r['clean'] / na)}) | {r['contradicted']} ({pct(r['contradicted'] / na)}) |")
    tot_f = sum(r["fine"] for r in M["sentences"].values())
    tot_c = sum(r["contradicted"] for r in M["sentences"].values())
    L += ["", f"The guarantee bounds one cell: shipped unchanged with a contradicted sentence <= {M['alpha']} of all answers.", "",
          f"## Sentence filter matrix ({ns} test sentences, alpha = {M['alpha']}; applied to every answer)", "",
          "| filter | not contradicted | contradicted |", "|---|---|---|"]
    for name in ("kept", "dropped"):
        r = M["sentences"][name]
        L.append(f"| {name} | {r['fine']} ({pct(r['fine'] / max(1, tot_f))} of these) | "
                 f"{r['contradicted']} ({pct(r['contradicted'] / max(1, tot_c))} of these) |")
    L += ["",
          "## Combiner (logistic regression, fit on train only)", "",
          "| feature | weight (standardised) |", "|---|---|"]
    for f, w in zip(th["features"], th["coef"]):
        L.append(f"| {f} | {w:+.3f} |")
    dest.write_text("\n".join(L) + "\n", encoding="utf-8")


def make_figures(S, covs, n_c, n_t, k, figs, beta_dist, betabinom) -> None:
    import logging

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # quiet if Arial is missing (Linux)
    plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]

    A = MAIN_ALPHA
    ba = S["resplits"]["by_alpha"]
    al = [r["alpha"] for r in ba]

    # 1. coverage vs target: average over the re-splits (band = middle 90%) and the single test split
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    tgt = [1 - a for a in al]
    lo, hi = [r["coverage"]["p05"] for r in ba], [r["coverage"]["p95"] for r in ba]
    mean = [r["coverage"]["mean"] for r in ba]
    test = [S["conformal"][str(a)] for a in al]
    ylo = min(min(lo), min(t["ci95"][0] for t in test), S["resplits"]["raw_mean"]) - 0.01
    ax.plot([ylo, 1], [ylo, 1], ls=(0, (4, 3)), color=INK_2, lw=1, label="target 1 − α")
    ax.axhline(S["resplits"]["raw_mean"], color=MUTED, lw=1.2, ls=":", label="unfiltered answers")
    ax.fill_between(tgt, lo, hi, color=BLUE, alpha=0.15, lw=0)
    ax.plot(tgt, mean, "-o", color=BLUE, lw=2, ms=5, label=f"average of {S['resplits']['n']:,} re-splits (band: middle 90%)")
    ax.errorbar(tgt, [t["coverage"] for t in test],
                yerr=[[t["coverage"] - t["ci95"][0] for t in test], [t["ci95"][1] - t["coverage"] for t in test]],
                fmt="o", ms=7, color=ORANGE, ecolor=ORANGE, elinewidth=1, capsize=0, mec=SURFACE, mew=1.5,
                label="this test split (95% CI)")
    ax.set_xlim(min(tgt) - 0.008, max(tgt) + 0.008)
    ax.set_ylim(ylo, 1.006)
    ax.set_xlabel("target coverage 1 − α")
    ax.set_ylabel("answers with no contradicted sentence")
    _style(ax, "Coverage meets the target at every α")
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    fig.savefig(figs / "coverage_vs_target.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 2. what the guarantee costs (re-split averages)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    fig.patch.set_facecolor(SURFACE)
    kept = [r["kept"]["mean"] for r in ba]
    ax.fill_between(al, [r["kept"]["p05"] for r in ba], [r["kept"]["p95"] for r in ba], color=BLUE, alpha=0.15, lw=0)
    ax.plot(al, kept, "-o", color=BLUE, lw=2, ms=5, label="sentences kept")
    nonempty = [r["nonempty"] for r in ba]
    ax.plot(al, nonempty, "-s", color=ORANGE, lw=2, ms=5, label="answers with at least one sentence kept")
    ax.axvline(A, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(A, 0.03, f" main α = {A:g}", color=INK_2, fontsize=8, va="bottom")
    ax.set_xlim(0, max(al) * 1.05)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("α (allowed share of answers with a contradicted sentence)")
    _style(ax, "What the guarantee costs (average of re-splits)")
    _legend(ax, loc="center right")
    fig.tight_layout()
    fig.savefig(figs / "retention_vs_alpha.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 3. re-split histogram with the Beta-binomial prediction
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    fig.patch.set_facecolor(SURFACE)
    bins = np.arange(n_t + 2) / n_t - 0.5 / n_t
    ax.hist(covs, bins=bins, color=BLUE, alpha=0.85, edgecolor=SURFACE, lw=0.4, label=f"{len(covs):,} re-splits")
    xs = np.arange(n_t + 1)
    ax.plot(xs / n_t, betabinom.pmf(xs, n_t, k, n_c + 1 - k) * len(covs), color=ORANGE, lw=2,
            label="conformal theory (Beta-binomial)")
    ax.axvline(1 - A, color=INK_2, ls=(0, (4, 3)), lw=1, label="target 1 − α")
    rs = S["resplits"]
    ax.text(0.02, 0.97, f"mean {rs['mean']:.1%} · theory {rs['theory_mean']:.1%}", transform=ax.transAxes,
            va="top", fontsize=8.5, color=INK)
    ax.set_xlim(max(0.0, float(np.quantile(covs, 0.001)) - 0.02), 1.0)
    ax.set_xlabel("test coverage")
    ax.set_ylabel("re-splits")
    _style(ax, f"Coverage across {len(covs):,} random re-splits (α = {A:g})")
    _legend(ax, loc="upper left", bbox_to_anchor=(0, 0.9))
    fig.tight_layout()
    fig.savefig(figs / "resplits.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 4. which scores matter: retention (re-split average) and AUROC, as two panels
    order = ["combined", "evidence", "agreement", "random"]
    ret = S["resplits"]["retention"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(order))[::-1]
    colors = [BLUE if n == "combined" else LIGHT for n in order]
    m = [ret[n]["mean"] for n in order]
    a1.barh(y, m, height=0.5, color=colors)
    a1.errorbar(m, y, xerr=[[mi - ret[n]["p05"] for mi, n in zip(m, order)],
                           [ret[n]["p95"] - mi for mi, n in zip(m, order)]],
                fmt="none", ecolor=INK_2, elinewidth=1, capsize=0)
    for yi, mi, n in zip(y, m, order):
        a1.text(0.01, yi, f"{mi:.0%}", va="center", fontsize=8.5,
                color=SURFACE if (n == "combined" and mi > 0.1) else INK)
    a1.set_yticks(y, order)
    a1.set_xlim(0, 1)
    a1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(a1, f"Sentences kept at α = {A:g} (re-split average)")
    a1.xaxis.grid(True, color=GRID, lw=0.8)
    a1.yaxis.grid(False)
    au = [S["ablation"][n]["auroc"] or 0.5 for n in order]
    a2.barh(y, au, height=0.5, color=colors)
    for yi, v in zip(y, au):
        a2.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    a2.axvline(0.5, color=INK_2, ls=(0, (4, 3)), lw=1)
    a2.text(0.5, y.max() + 0.4, " chance", fontsize=8, color=INK_2, va="bottom")
    a2.set_xlim(0, 1)
    _style(a2, "AUROC on test sentences")
    a2.xaxis.grid(True, color=GRID, lw=0.8)
    a2.yaxis.grid(False)
    fig.text(0.01, 0.01, "Coverage is guaranteed whichever score is used; a better score only changes how much is kept.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figs / "ablation.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    matrix_figure(S["matrices"], figs / "matrices.png", plt)


# one-hue sequential ramp (blue 100 -> 700) and ink from the dataviz reference palette
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# palette: United Nations Blue, Black Chocolate, Android Green, Medium Spring Bud, Axolotl (series steps deepened
# just enough to pass the contrast and colour-blindness checks against the page surface)
INK, INK_2, SURFACE = "#1D3309", "#4E6238", "#FBFCF7"
BLUE, ORANGE = "#4F86D4", "#7E9435"                      # series 1 (blue) and 2 (green)
MUTED, GRID, AXIS, LIGHT = "#6A804F", "#E2E8D2", "#C9D3B3", "#C9D3B3"


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8.5, color=AXIS)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def _legend(ax, **kw) -> None:
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, **kw)


def _grid(ax, values, shares, row_labels, col_labels, annot, title, highlight=None):
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    nr, nc = len(row_labels), len(col_labels)
    for i in range(nr):
        for j in range(nc):
            v = float(shares[i][j])
            ax.add_patch(Rectangle((j, nr - 1 - i), 1, 1, facecolor=cmap(v), edgecolor=SURFACE, lw=2))
            ax.text(j + 0.5, nr - 1 - i + 0.5, annot(i, j), ha="center", va="center", fontsize=9,
                    color=SURFACE if v > 0.55 else INK)
    if highlight is not None:
        i, j = highlight
        ax.add_patch(Rectangle((j + 0.04, nr - 1 - i + 0.04), 0.92, 0.92, fill=False, edgecolor=INK, lw=1.6,
                               linestyle=(0, (4, 2))))
    ax.set_xlim(0, nc)
    ax.set_ylim(0, nr)
    ax.set_xticks([j + 0.5 for j in range(nc)], col_labels, fontsize=8.5, color=INK_2)
    ax.set_yticks([nr - 1 - i + 0.5 for i in range(nr)], row_labels, fontsize=8.5, color=INK_2)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=10, color=INK, pad=26)


def matrix_figure(M: dict, dest, plt) -> None:
    """Left: decision x truth for test answers. Right: filter x label for test sentences."""
    na = M["n_answers"]
    dec_rows = [("ship", "shipped\nunchanged"), ("review", "filtered,\nsent for review"), ("block", "blocked")]
    A = [[M["answers"][d]["clean"], M["answers"][d]["contradicted"]] for d, _ in dec_rows]
    A_share = [[x / na for x in row] for row in A]
    amax = max(max(r) for r in A_share) or 1
    sent = [M["sentences"]["kept"], M["sentences"]["dropped"]]
    cols = [sum(r["fine"] for r in sent), sum(r["contradicted"] for r in sent)]
    B = [[r["fine"], r["contradicted"]] for r in sent]
    B_share = [[B[i][j] / max(1, cols[j]) for j in range(2)] for i in range(2)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor(SURFACE)
    _grid(a1, A, [[v / amax for v in row] for row in A_share], [lab for _, lab in dec_rows],
          ["no contradicted\nsentence", "has a contradicted\nsentence"],
          lambda i, j: f"{A[i][j]}\n{A_share[i][j]:.1%} of answers",
          f"What happened to each test answer (α = {M['alpha']:g})", highlight=(0, 1))
    a1.text(1.0, -0.12, f"dashed cell is the bounded risk: ≤ {M['alpha']:.0%} of all answers",
            transform=a1.transAxes, ha="right", va="top", fontsize=8, color=INK_2)
    _grid(a2, B, B_share, ["kept", "dropped"], ["not\ncontradicted", "contradicted"],
          lambda i, j: f"{B[i][j]}\n{B_share[i][j]:.0%} of column",
          "What the sentence filter did to each test sentence")
    fig.tight_layout()
    fig.savefig(dest, dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
