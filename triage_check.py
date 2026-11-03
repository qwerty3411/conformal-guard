"""
Run:  python triage_check.py
Writes results/triage_check.md, results/triage_check.json and results/figures/triage_check.png.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

import calibrate as cb
from cg_common import ROOT

BUDGETS = (0.10, 0.20, 0.30, 0.50)
N_BOOT = 2000


def answers(part: pd.DataFrame) -> pd.DataFrame:
    """One row per answer: weakest and strongest sentence score, r (for tau), contradicted or not."""
    g = part.assign(_fs=np.where(part["is_false"], part["combined"], -np.inf)).groupby("id")
    return pd.DataFrame({"a": g["combined"].min(), "top": g["combined"].max(), "r": g["_fs"].max(),
                         "wrong": g["is_false"].any(), "split": g["split"].first()})


def rule(a_cal: np.ndarray, wrong_cal: np.ndarray, r_cal: np.ndarray, alpha: float) -> tuple[float, float]:
    """Ship threshold (CRC on P(shipped and contradicted)) and sentence threshold, as in calibrate.py."""
    grid = np.unique(np.concatenate([a_cal, [np.inf]]))
    wa = np.sort(a_cal[wrong_cal])
    risks = (len(wa) - np.searchsorted(wa, grid, side="left")) / len(a_cal)   # mean((a >= l) & wrong)
    return cb.crc_threshold(risks, grid, len(a_cal), alpha), cb.conformal_tau(r_cal, alpha)


def judge(lam: float, tau: float, a, top, wrong) -> dict:
    ship = a >= lam
    block = ~ship & (top <= tau)                 # no sentence survives the filter
    review = ~ship & ~block
    return {"answers": int(len(a)), "contradicted": int(wrong.sum()),
            "ship": int(ship.sum()), "review": int(review.sum()), "block": int(block.sum()),
            "ship_contradicted": int((ship & wrong).sum()), "review_contradicted": int((review & wrong).sum()),
            "block_contradicted": int((block & wrong).sum())}


def describe(c: dict) -> dict:
    from scipy.stats import fisher_exact
    rw, rn = c["review_contradicted"], c["review"] - c["review_contradicted"]
    ow, on = c["contradicted"] - rw, (c["answers"] - c["review"]) - (c["contradicted"] - rw)
    lo, hi = cb.clopper_pearson(rw, c["contradicted"]) if c["contradicted"] else (None, None)
    rr = (rw / max(1, c["review"])) / (ow / max(1, c["answers"] - c["review"])) if ow else None
    return {**c, "review_share": c["review"] / c["answers"],
            "catch": rw / c["contradicted"] if c["contradicted"] else None, "catch_ci95": [lo, hi],
            "shipped_and_contradicted": c["ship_contradicted"] / c["answers"],
            "relative_risk": rr, "fisher_p": float(fisher_exact([[rw, rn], [ow, on]])[1])}


def auroc_ci(wrong: np.ndarray, a: np.ndarray, rng) -> list[float]:
    from sklearn.metrics import roc_auc_score
    point = float(roc_auc_score(wrong, -a))      # lower weakest-sentence score = riskier
    boots, n = [], len(a)
    while len(boots) < N_BOOT:
        i = rng.integers(0, n, n)
        if 0 < wrong[i].sum() < n:
            boots.append(roc_auc_score(wrong[i], -a[i]))
    return [point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def budgets(a: np.ndarray, wrong: np.ndarray) -> list[dict]:
    order = np.argsort(a, kind="mergesort")
    out = []
    for b in BUDGETS:
        k = int(round(b * len(a)))
        x = int(wrong[order[:k]].sum())
        lo, hi = cb.clopper_pearson(x, int(wrong.sum()))
        out.append({"budget": b, "reviewed": k, "caught": x, "of": int(wrong.sum()),
                    "catch": x / wrong.sum(), "ci95": [lo, hi]})
    return out


def pct(x, d=0) -> str:
    return "–" if x is None else f"{100 * x:.{d}f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--false-label", choices=["contradicted", "strict"], default="contradicted")
    ap.add_argument("--labels", choices=["applied", "v2"], default="applied")
    ap.add_argument("--alpha", type=float, default=cb.MAIN_ALPHA)
    args = ap.parse_args()
    alpha = args.alpha
    rng = np.random.default_rng(cb.SEED + 7)

    df = cb.load(args.false_label, args.labels)
    model = cb.fit_combiner(df[df["split"] == "train"])
    df["combined"] = model.predict_proba(df[cb.FEATURES].to_numpy())[:, 1]
    A = answers(df[df["split"].isin(["calibrate", "test"])])
    a, top, r, wrong = (A[c].to_numpy() for c in ("a", "top", "r", "wrong"))
    is_cal = (A["split"] == "calibrate").to_numpy()
    n_c, n_t = int(is_cal.sum()), int((~is_cal).sum())

    # 1. both directions of the frozen split
    lam_f, tau_f = rule(a[is_cal], wrong[is_cal], r[is_cal], alpha)
    fwd = judge(lam_f, tau_f, a[~is_cal], top[~is_cal], wrong[~is_cal])
    lam_s, tau_s = rule(a[~is_cal], wrong[~is_cal], r[~is_cal], alpha)
    swp = judge(lam_s, tau_s, a[is_cal], top[is_cal], wrong[is_cal])
    pooled = {k: fwd[k] + swp[k] for k in fwd}
    S = {"alpha": alpha, "n_calibrate": n_c, "n_test": n_t,
         "forward": {"ship_lambda": lam_f, "tau": tau_f, **describe(fwd)},
         "swapped": {"ship_lambda": lam_s, "tau": tau_s, **describe(swp)},
         "pooled": describe(pooled)}

    # 2. random re-splits at the original sizes
    rows = []
    for _ in range(cb.N_RESPLITS):
        p = rng.permutation(len(a))
        ci, ti = p[:n_c], p[n_c:n_c + n_t]
        lam, tau = rule(a[ci], wrong[ci], r[ci], alpha)
        c = judge(lam, tau, a[ti], top[ti], wrong[ti])
        rows.append([c["review_contradicted"] / max(1, c["contradicted"]), c["review"] / c["answers"],
                     c["ship_contradicted"] / c["answers"], c["contradicted"]])
    R = np.array(rows)

    def band(v):
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p025": float(np.quantile(v, 0.025)), "p975": float(np.quantile(v, 0.975))}
    S["resplits"] = {"n": cb.N_RESPLITS, "catch": band(R[:, 0]), "review_share": band(R[:, 1]),
                     "shipped_and_contradicted": band(R[:, 2]), "contradicted_per_test_part": band(R[:, 3]),
                     "share_with_catch_above_review_share": float((R[:, 0] > R[:, 1]).mean())}

    # 3. review budgets and answer-level AUROC: all answers, and the test split alone
    S["budgets"] = {"all": budgets(a, wrong), "test": budgets(a[~is_cal], wrong[~is_cal])}
    S["auroc_weakest_sentence"] = {"all": auroc_ci(wrong, a, rng), "test": auroc_ci(wrong[~is_cal], a[~is_cal], rng)}

    out = ROOT / ("results" if args.false_label == "contradicted" else "results/strict")
    if args.labels == "v2":
        out = ROOT / "results" / "v2_labels"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "triage_check.json").write_text(json.dumps(S, indent=1, default=float))
    md = markdown(S)
    (out / "triage_check.md").write_text(md, encoding="utf-8")
    figure(S, a, wrong, out / "figures" / "triage_check.png")
    print(md)
    print(f"written to {out / 'triage_check.md'}, triage_check.json, figures/triage_check.png")


def markdown(S: dict) -> str:
    f, s, p, rs = S["forward"], S["swapped"], S["pooled"], S["resplits"]

    def row(name, d):
        lo, hi = d["catch_ci95"]
        rr = "–" if d["relative_risk"] is None else f"{d['relative_risk']:.1f}"
        return (f"| {name} | {d['answers']} | {d['contradicted']} | {pct(d['review_share'])} | "
                f"{d['review_contradicted']} of {d['contradicted']} = {pct(d['catch'])} ({pct(lo)}–{pct(hi)}) | "
                f"{d['block_contradicted']} | {d['ship_contradicted']} = {pct(d['shipped_and_contradicted'], 1)} | "
                f"{rr} | {d['fisher_p']:.2g} |")
    L = ["# Triage check: does the review queue hold up on more answers?", "",
         f"Robustness check on the frozen data (alpha = {S['alpha']:g}); nothing re-tuned, headline results unchanged. "
         "The combiner is fitted on the train split only, so calibration and test answers are both unseen by it.", "",
         "## 1. Both directions of the frozen split", "",
         "| Judged answers | Answers | Contradicted | Sent to review | Contradicted answers in the review queue (95% CI) "
         "| Blocked & contradicted | Shipped & contradicted | Relative risk | Fisher p |",
         "|---|---|---|---|---|---|---|---|---|",
         row("Test (calibrated on calibration answers; the headline)", f),
         row("Calibration (calibrated on test answers; swapped)", s),
         row("**Pooled: every answer judged once**", p), "",
         f"Ship threshold {f['ship_lambda']:.5f} forward, {s['ship_lambda']:.5f} swapped.", "",
         f"## 2. {rs['n']:,} random re-splits ({S['n_calibrate']} calibrate / {S['n_test']} judged)", "",
         "| | Mean | Median | Middle 95% of re-splits |", "|---|---|---|---|"]
    for k, name in (("catch", "Contradicted answers in the review queue"), ("review_share", "Answers sent to review"),
                    ("shipped_and_contradicted", f"Shipped with a contradiction (bounded by {S['alpha']:g} on average)"),
                    ("contradicted_per_test_part", "Contradicted answers per judged part")):
        b, d = rs[k], 0 if k == "contradicted_per_test_part" else 1
        if k == "contradicted_per_test_part":
            L.append(f"| {name} | {b['mean']:.1f} | {b['median']:.0f} | {b['p025']:.0f}–{b['p975']:.0f} |")
        else:
            L.append(f"| {name} | {pct(b['mean'], d)} | {pct(b['median'], d)} | {pct(b['p025'], d)}–{pct(b['p975'], d)} |")
    L += ["", f"In {pct(rs['share_with_catch_above_review_share'])} of re-splits the queue held a larger share of the "
          "contradicted answers than of all answers (random review would hold the same share).", "",
          "## 3. Review budgets: send the riskiest share by weakest-sentence score", "",
          "| Reviewed | All answers: contradicted caught (95% CI) | Test split only |", "|---|---|---|"]
    for ba, bt in zip(S["budgets"]["all"], S["budgets"]["test"]):
        L.append(f"| {pct(ba['budget'])} | {ba['caught']} of {ba['of']} = {pct(ba['catch'])} "
                 f"({pct(ba['ci95'][0])}–{pct(ba['ci95'][1])}) | {bt['caught']} of {bt['of']} = {pct(bt['catch'])} |")
    au, at = S["auroc_weakest_sentence"]["all"], S["auroc_weakest_sentence"]["test"]
    L += ["", f"Answer-level AUROC of the weakest-sentence score: all answers {au[0]:.2f} "
          f"(95% bootstrap CI {au[1]:.2f}–{au[2]:.2f}); test split {at[0]:.2f} ({at[1]:.2f}–{at[2]:.2f}).", ""]
    return "\n".join(L)


def figure(S: dict, a: np.ndarray, wrong: np.ndarray, dest) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def curve(a_, w_):
        o = np.argsort(a_, kind="mergesort")
        x = np.arange(len(a_) + 1) / len(a_)
        y = np.concatenate([[0], np.cumsum(w_[o])]) / w_.sum()
        return x, y
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=cb.SURFACE)
    ax.plot([0, 1], [0, 1], color=cb.MUTED, lw=1.1, ls=":", label="random order")
    x, y = curve(a, wrong)
    p = S["pooled"]
    ax.step(x, y, where="post", color=cb.BLUE, lw=2,
            label=f"all {p['answers']} answers ({p['contradicted']} contradicted)")
    ax.errorbar([p["review_share"]], [p["catch"]],
                yerr=[[p["catch"] - p["catch_ci95"][0]], [p["catch_ci95"][1] - p["catch"]]],
                fmt="o", color=cb.ORANGE, ms=7, capsize=3, lw=1.4,
                label=f"the ship rule, every answer judged once: {p['review_contradicted']} of {p['contradicted']}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("drafts the pharmacist reviews (riskiest first)")
    ax.set_ylabel("contradicted answers she sees")
    cb._style(ax, "Contradicted answers caught by reviewing the riskiest drafts")
    cb._legend(ax, loc="lower right")
    fig.tight_layout()
    fig.savefig(dest, dpi=160, facecolor=cb.SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
