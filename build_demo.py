"""
 python build_demo.py                # writes docs/data.js and docs/figures/, then open docs/index.html
python build_demo.py --no-domain    # skip the domain check (no embedding model needed)

Inputs:  results/thresholds.json, results/summary.json, results/test_claims.parquet, results/figures/,
         data/questions_final.jsonl, data/scores.parquet, cache/generations/<id>/0.json,
         data/raw (doctor-written reference answers), optional: data/contradiction_details.parquet,
         results/label_audit.csv, results/v2_labels/summary.json
Output:  docs/data.js (window.CG_DATA = {...}), docs/figures/*.png. docs/index.html is the page itself.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time

import numpy as np

from cg_common import ROOT, load_config, path

DEMO = ROOT / "docs"
FIGURES = ["coverage_vs_target", "resplits", "retention_vs_alpha", "matrices", "ablation"]


def clean(x):
    """JSON-safe: infinities become the strings '-inf'/'inf', NaN becomes null, numpy types become Python."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return round(x, 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def load_reference(cfg: dict) -> dict[str, str]:
    from prepare_data import fetch

    out: dict[str, str] = {}
    for group in cfg["data"]["groups"].values():
        for src in group:
            for r in fetch(cfg, src["file"]):
                if r.get("Free_form_answer"):
                    out[r["Question"].strip()] = r["Free_form_answer"].strip()
    return out


def domain_scores(cfg: dict, questions: list[dict], fake: bool) -> tuple[dict[str, float], float]:
    """Same check as guard.py: similarity to the nearest calibration question vs the 1% calibration quantile."""
    from build_index import Embedder
    from guard import DOMAIN_QUANTILE

    emb = Embedder(cfg["evidence"]["embedding_model"], fake=fake)
    cal = [q["question"] for q in questions if q["split"] == "calibrate"]
    test = [q for q in questions if q["split"] == "test"]
    ce = emb(cal)
    sims = ce @ ce.T
    np.fill_diagonal(sims, -1.0)
    bar = float(np.quantile(sims.max(axis=1), DOMAIN_QUANTILE))
    te = emb([q["question"] for q in test])
    best = (te @ ce.T).max(axis=1)
    return {q["id"]: float(s) for q, s in zip(test, best)}, bar


def audit_summary(cfg: dict, rule: str) -> dict | None:
    """Label-audit precision for the applied rule and for v2, from results/label_audit.csv."""
    import pandas as pd

    import audit_labels as al

    if not (al.DETAILS.exists() and al.AUDIT.exists()):
        return None
    d = pd.read_parquet(al.DETAILS)
    flags = al.rule_flags(d, cfg)
    verdicts = al.read_audit()
    v = np.array([verdicts.get((a, int(b))) for a, b in zip(d["id"], d["idx"])], dtype=object)
    judged = np.isin(v, ["r", "f"])
    out = {"rule": rule}
    for name in dict.fromkeys(["v2", rule]):
        if name not in flags:
            continue
        m = flags[name].to_numpy() & judged
        out[name] = {"audited": int(m.sum()), "real": int((m & (v == "r")).sum()),
                     "flags_total": int(flags[name].sum())}
    ctrl = judged & ~d["flag_v2"].to_numpy()
    out["controls"] = {"audited": int(ctrl.sum()), "real": int((ctrl & (v == "r")).sum())}
    return out


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-domain", action="store_true", help="skip the domain-similarity check")
    ap.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    t0 = time.monotonic()
    cfg = load_config()
    res = ROOT / "results"
    for f in ("thresholds.json", "summary.json", "test_claims.parquet"):
        if not (res / f).exists():
            raise SystemExit(f"results/{f} not found: run `python calibrate.py --alpha <main alpha>` first")
    th = json.loads((res / "thresholds.json").read_text(encoding="utf-8"))
    S = json.loads((res / "summary.json").read_text(encoding="utf-8"))
    tc = pd.read_parquet(res / "test_claims.parquet")
    sc = pd.read_parquet(ROOT / "data" / "scores.parquet", columns=["id", "idx", "evidence_urls", "retrieval_sim"])
    tc = tc.merge(sc, on=["id", "idx"], how="left")
    questions = [json.loads(l) for l in (ROOT / "data" / "questions_final.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    qby = {q["id"]: q for q in questions}
    titles = {}
    for line in (ROOT / cfg["evidence"]["dir"] / "passages.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            titles.setdefault(p["url"], p["title"])
    reference = load_reference(cfg)
    conflicts = {}
    det_path = ROOT / "data" / "contradiction_details.parquet"
    if det_path.exists():
        det = pd.read_parquet(det_path, columns=["id", "idx", "best_window"])
        conflicts = {(a, int(b)): w for a, b, w in det.itertuples(index=False)}
    if args.no_domain:
        dom, bar = {}, None
    else:
        dom, bar = domain_scores(cfg, questions, args.fake)
    gen_dir = path(cfg, "cache")

    answers = []
    for qid, rows in tc.groupby("id", sort=False):
        q = qby[qid]
        rows = rows.sort_values("idx")
        text = json.loads((gen_dir / qid / "0.json").read_text(encoding="utf-8"))["content"].strip()
        claims = []
        for _, r in rows.iterrows():
            urls = json.loads(r["evidence_urls"]) if isinstance(r["evidence_urls"], str) else []
            c = {"i": int(r["idx"]), "text": r["text"], "s": int(r["start"]), "e": int(r["end"]),
                 "score": float(r["combined"]), "label": r["label"],
                 "agree": float(r["agree_mean"]), "evidence": float(r["evidence_support"]),
                 "links": [{"url": u, "title": titles.get(u, u)} for u in urls]}
            if r["label"] == "contradicted" and (qid, int(r["idx"])) in conflicts:
                c["conflict"] = conflicts[(qid, int(r["idx"]))]
            claims.append(c)
        ref = reference.get(q["question"].strip()) or " ".join(q.get("must_have", []))
        answers.append({"id": qid, "group": q["group"], "source": q.get("source"), "question": q["question"],
                        "answer": text, "reference": ref, "claims": claims,
                        "domain": dom.get(qid)})

    v2 = None
    v2_path = res / "v2_labels" / "summary.json"
    if v2_path.exists():
        V = json.loads(v2_path.read_text(encoding="utf-8"))
        a = str(V["main_alpha"])
        v2 = {"alpha": V["main_alpha"], "raw_wrong": V["raw"]["test_answers_with_false_claim"],
              "coverage_test": V["conformal"][a]["coverage"], "resplit_mean": V["resplits"]["mean"],
              "theory": V["resplits"]["theory_mean"], "kept": V["resplits"]["retention"]}

    data = {
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "model": cfg["api"]["model"], "generated": "2026-10-26",
        "label_rule": th.get("label_rule"), "main_alpha": th["main_alpha"],
        "ship_lambda": th["ship_lambda"], "n_cal": th["n_calibration_answers"],
        "tau": [{"alpha": float(a), "tau": t} for a, t in sorted(th["tau"].items(), key=lambda kv: float(kv[0]))],
        "domain_bar": bar,
        "summary": {"counts": S["counts"], "raw": S["raw"], "conformal": S["conformal"],
                    "resplits": {k: S["resplits"][k] for k in ("alpha", "n", "mean", "p05", "p95", "theory_mean",
                                                               "retention", "raw_mean", "by_alpha")},
                    "abstention": S["abstention"], "matrices": S["matrices"],
                    "auroc": {k: v["auroc"] for k, v in S["ablation"].items()}},
        "audit": audit_summary(cfg, th.get("label_rule", "v2")),
        "v2": v2,
        "answers": answers,
    }
    DEMO.mkdir(exist_ok=True)
    js = "window.CG_DATA = " + json.dumps(clean(data), ensure_ascii=False, separators=(",", ":")) + ";\n"
    (DEMO / "data.js").write_text(js, encoding="utf-8")
    (DEMO / "figures").mkdir(exist_ok=True)
    copied = 0
    for f in FIGURES:
        src = res / "figures" / f"{f}.png"
        if src.exists():
            shutil.copyfile(src, DEMO / "figures" / f"{f}.png")
            copied += 1
    n_claims = sum(len(a["claims"]) for a in answers)
    print(f"wrote docs/data.js ({len(js) / 1e6:.1f} MB): {len(answers)} test answers, {n_claims} sentences; "
          f"{copied} figures -> docs/figures/ ({time.monotonic() - t0:.0f} s)")
    if not (DEMO / "index.html").exists():
        print("docs/index.html is missing: put the page file in the docs folder")
    print("open docs/index.html in a browser (double-click works; no server needed)")


if __name__ == "__main__":
    main()
