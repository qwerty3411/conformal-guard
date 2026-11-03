"""
python audit_labels.py                 # read flagged claims one at a time and judge them (about 20 min for 40)
python audit_labels.py --report        # how many flags each rule keeps, and how many of them are real
python audit_labels.py --apply local   # switch data/claims.parquet to that rule 

Rules: 
  v2         the doctor's answer, or all reference facts together, contradict the sentence (p >= 0.7)
  both       every available reference (all facts together AND the doctor's answer) contradicts it (p >= 0.7)
  strict     as v2, with p >= 0.9
  local      v2, and one specific 2-sentence passage of the reference contradicts it (p >= 0.9),
             and no passage supports it
  symmetric  local, and the contradiction still holds with the passage and the sentence swapped (p >= 0.5)
A v2 flag that a rule drops becomes "unverifiable". Nothing else changes.

Clean protocol: audit uses claims from the train split only, a few unflagged claims are mixed in so user
don't know which ones the model flagged, and the report shows label quality only (never how labels relate
to the scores). Thuerefore choosing a rule can't tune the guarantee. Run this before calibrate.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import textwrap
import time

from cg_common import ROOT, load_config, write_json_atomic
import label_claims as _lc
from label_claims import NLI, split_sentences

LABEL_VERSION = getattr(_lc, "LABEL_VERSION", 2)   # older label_claims.py files don't define it


def load_freeform(cfg: dict) -> dict[str, str]:
    """MedLFQA's doctor-written answer for each question, keyed by question text."""
    from prepare_data import fetch

    out: dict[str, str] = {}
    for group in cfg["data"]["groups"].values():
        for src in group:
            for r in fetch(cfg, src["file"]):
                if r.get("Free_form_answer"):
                    out[r["Question"].strip()] = r["Free_form_answer"].strip()
    return out

CLAIMS = ROOT / "data" / "claims.parquet"
BACKUP = ROOT / "data" / "claims_v2.parquet"           # untouched v2 labels; every rule starts from these
DETAILS = ROOT / "data" / "contradiction_details.parquet"
AUDIT = ROOT / "results" / "label_audit.csv"
RULE_FILE = ROOT / "data" / "label_rule.json"
RULES = ["v2", "both", "strict", "local", "symmetric"]
STRONG, REVERSE, WINDOW = 0.9, 0.5, 2
SEED = 20261101
CONTROL_EVERY = 5               # every 5th item is an unflagged claim (blinding + a rough miss rate)
DETAILS_VERSION = 1


# ----------------------------------------------------------------------------- features
def windows(text: str, size: int = WINDOW) -> list[str]:
    sents = [s for s, _, _ in split_sentences(text, min_words=1)]
    if len(sents) <= size:
        return [" ".join(sents)] if sents else []
    return [" ".join(sents[i:i + size]) for i in range(len(sents) - size + 1)]


def compute_details(cfg: dict, base, questions: dict, fake: bool):
    """Per-reference NLI for every v2-flagged claim, plus a sample of unflagged train claims (controls)."""
    import pandas as pd

    flagged = base[base["label"] == "contradicted"]
    rng = random.Random(SEED)
    unflagged_train = base[(base["label"] != "contradicted") & (base["split"] == "train")]
    n_ctrl = min(len(unflagged_train), max(10, len(flagged[flagged["split"] == "train"]) // (CONTROL_EVERY - 1) + 5))
    ctrl_idx = rng.sample(sorted(unflagged_train.index), n_ctrl)
    rows = pd.concat([flagged, base.loc[ctrl_idx]])
    freeform = load_freeform(cfg)
    nli = NLI(cfg["nli"]["model"], cfg["nli"]["batch_size"], cfg["nli"]["max_length"], fake=fake)

    plan, pairs = [], []
    for ix, r in rows.iterrows():
        q = questions[r["id"]]
        facts = [s.strip() for s in q["must_have"] + q["nice_to_have"] if s.strip()]
        doc = freeform.get(q["question"].strip())
        wins = windows(doc) if doc else windows(" ".join(facts))
        start = len(pairs)
        if facts:
            pairs.append((" ".join(facts), r["text"]))
        if doc:
            pairs.append((doc, r["text"]))
        pairs += [(w, r["text"]) for w in wins]
        plan.append((ix, start, bool(facts), bool(doc), wins))
    t0 = time.monotonic()
    probs = nli(pairs)

    recs, rev_pairs = [], []
    for ix, start, has_f, has_d, wins in plan:
        p = probs[start:start + has_f + has_d + len(wins)]
        con_facts = p[0][1] if has_f else None
        con_doc = p[int(has_f)][1] if has_d else None
        wp = p[has_f + has_d:]
        best = max(range(len(wp)), key=lambda i: wp[i][1]) if wp else None
        r = rows.loc[ix]
        recs.append({"id": r["id"], "idx": int(r["idx"]), "split": r["split"], "group": r["group"],
                     "text": r["text"], "flag_v2": r["label"] == "contradicted",
                     "con_facts": con_facts, "con_doc": con_doc,
                     "con_win": wp[best][1] if wp else 0.0, "ent_win": max(x[0] for x in wp) if wp else 0.0,
                     "best_window": wins[best] if wp else ""})
        rev_pairs.append((r["text"], wins[best]) if wp else None)
    rp = nli([x for x in rev_pairs if x]) if any(rev_pairs) else []
    it = iter(rp)
    for rec, x in zip(recs, rev_pairs):
        rec["con_rev"] = next(it)[1] if x else 0.0
    print(f"NLI details for {len(recs)} claims ({len(pairs) + len(rp)} pairs) in {time.monotonic() - t0:.0f} s")
    return pd.DataFrame(recs)


def rule_flags(d, cfg: dict):
    """Boolean flag per rule for each row of the details table."""
    import pandas as pd

    t = cfg["nli"]["contradict_threshold"]
    cf, cd = d["con_facts"].astype(float), d["con_doc"].astype(float)
    ctx_max = pd.concat([cf, cd], axis=1).max(axis=1, skipna=True).fillna(0.0)
    both = (cf.isna() | (cf >= t)) & (cd.isna() | (cd >= t)) & ~(cf.isna() & cd.isna())
    v2 = d["flag_v2"].astype(bool)
    local = v2 & (d["con_win"] >= STRONG) & (d["ent_win"] < cfg["nli"]["entail_threshold"])
    return pd.DataFrame({"v2": v2, "both": v2 & both, "strict": v2 & (ctx_max >= STRONG),
                         "local": local, "symmetric": local & (d["con_rev"] >= REVERSE)}, index=d.index)


def details_key(cfg: dict, base) -> str:
    blob = {"nli": cfg["nli"]["model"], "max_length": cfg["nli"]["max_length"], "window": WINDOW,
            "v": DETAILS_VERSION, "label_version": LABEL_VERSION,
            "claims": hashlib.sha256("\n".join(f"{a}:{b}:{c}:{d}" for a, b, c, d in
                                               base[["id", "idx", "text", "label"]].itertuples(index=False))
                                     .encode()).hexdigest()}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]


def load_all(cfg: dict, fake: bool):
    import pandas as pd

    if not CLAIMS.exists():
        raise SystemExit("data/claims.parquet not found: run `python label_claims.py` first")
    current = pd.read_parquet(CLAIMS)
    if "label_rule" not in current.columns:          # fresh output of label_claims.py: that is the v2 base
        if not BACKUP.exists() or not pd.read_parquet(BACKUP).equals(current):
            shutil.copyfile(CLAIMS, BACKUP)
            print(f"saved the v2 labels from label_claims.py to {BACKUP.relative_to(ROOT)}")
    elif not BACKUP.exists():
        raise SystemExit("data/claims_v2.parquet is missing: run `python label_claims.py` to rebuild the v2 labels")
    base = pd.read_parquet(BACKUP)
    questions = {}
    for line in (ROOT / "data" / "questions_final.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            questions[q["id"]] = q
    key = details_key(cfg, base)
    if DETAILS.exists():
        d = pd.read_parquet(DETAILS)
        if len(d) and d["key"].iloc[0] == key:
            return base, d, questions
    d = compute_details(cfg, base, questions, fake)
    d["key"] = key
    d.to_parquet(DETAILS, index=False)
    return base, d, questions


# ----------------------------------------------------------------------------- audit
def read_audit() -> dict[tuple[str, int], str]:
    if not AUDIT.exists():
        return {}
    with AUDIT.open(encoding="utf-8-sig", newline="") as f:
        return {(r["id"], int(r["idx"])): r["verdict"] for r in csv.DictReader(f)}


def write_audit(rows: list[dict]) -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    tmp = AUDIT.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "idx", "split", "group", "verdict", "text", "judged_at"])
        w.writeheader()
        w.writerows(rows)
    tmp.replace(AUDIT)


def audit_order(d) -> list[int]:
    """Train-split items: flagged claims in a fixed random order, with an unflagged control every 5th slot."""
    tr = d[d["split"] == "train"]
    flagged = sorted(tr[tr["flag_v2"]].index, key=lambda i: (tr.at[i, "id"], tr.at[i, "idx"]))
    ctrl = sorted(tr[~tr["flag_v2"]].index, key=lambda i: (tr.at[i, "id"], tr.at[i, "idx"]))
    rng = random.Random(SEED)
    rng.shuffle(flagged)
    rng.shuffle(ctrl)
    order, ci = [], iter(ctrl)
    for i, ix in enumerate(flagged, 1):
        order.append(ix)
        if i % (CONTROL_EVERY - 1) == 0:
            nxt = next(ci, None)
            if nxt is not None:
                order.append(nxt)
    return order


def wrap(text: str, indent: str = "    ", width: int = 100) -> str:
    return textwrap.fill(" ".join(str(text).split()), width=width, initial_indent=indent, subsequent_indent=indent)


def run_audit(d, questions: dict, cfg: dict, target: int) -> None:
    freeform = load_freeform(cfg)
    done = read_audit()
    rows = []
    if AUDIT.exists():
        with AUDIT.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    order = audit_order(d)
    flag_keys = set(zip(d.loc[d["flag_v2"], "id"], d.loc[d["flag_v2"], "idx"].astype(int)))
    n_flag_done = sum(1 for ix in order if (d.at[ix, "id"], int(d.at[ix, "idx"])) in done and d.at[ix, "flag_v2"])
    print(textwrap.dedent(f"""
    For each item, read the SENTENCE and then the doctor-written REFERENCE. Decide only this:
    does the reference say something that makes the sentence wrong?
      r  real disagreement: the reference states something incompatible with the sentence
      f  no disagreement: same meaning, compatible extra detail, or about a different case/subtype
      u  unsure
      q  save and stop (run the command again to continue where you left off)
    Judge against the reference, not your own medical knowledge. Aim for {target} items; {n_flag_done} done so far.
    """))
    pos = 0
    for ix in order:
        r = d.loc[ix]
        k = (r["id"], int(r["idx"]))
        if k in done:
            continue
        pos += 1
        q = questions[r["id"]]
        facts = [s.strip() for s in q["must_have"] + q["nice_to_have"] if s.strip()]
        doc = freeform.get(q["question"].strip())
        print("=" * 100)
        print(f"item {len(done) + 1}  ·  {r['group']}  ·  {r['id']}")
        print("QUESTION:")
        print(wrap(q["question"]))
        print("SENTENCE (from the Muse Spark answer):")
        print(wrap(r["text"]))
        if r["best_window"]:
            print("PASSAGE TO COMPARE FIRST:")
            print(wrap(r["best_window"]))
        if doc:
            print("DOCTOR'S ANSWER (full):")
            print(wrap(doc[:2500] + (" ..." if len(doc) > 2500 else "")))
        if facts:
            print("REFERENCE FACTS:")
            for f_ in facts:
                print(wrap(f"- {f_}"))
        while True:
            ans = input("verdict [r/f/u/q] > ").strip().lower()
            if ans in {"r", "f", "u", "q"}:
                break
        if ans == "q":
            break
        done[k] = ans
        rows.append({"id": r["id"], "idx": int(r["idx"]), "split": r["split"], "group": r["group"],
                     "verdict": ans, "text": r["text"], "judged_at": time.strftime("%Y-%m-%d %H:%M")})
        write_audit(rows)
        if k in flag_keys and sum(1 for x in done if x in flag_keys) == target:
            print(f"\n{target} flagged items judged. You can stop here (q) or keep going for tighter numbers.")
    print(f"\nsaved {len(done)} verdicts to {AUDIT.relative_to(ROOT)}. Next: python audit_labels.py --report")


# ----------------------------------------------------------------------------- report / apply
def clopper_pearson(x: int, n: int) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    return lo, hi


def answer_rate(base, flagged_keys: set) -> float:
    k = list(zip(base["id"], base["idx"].astype(int)))
    has = base.assign(_f=[x in flagged_keys for x in k]).groupby("id")["_f"].any()
    return float(has.mean())


def report(base, d, cfg: dict) -> str | None:
    flags = rule_flags(d, cfg)
    verdicts = read_audit()
    d = d.assign(verdict=[verdicts.get((a, int(b))) for a, b in zip(d["id"], d["idx"])])
    judged = d["verdict"].isin(["r", "f"])
    n_flag_j = int((judged & d["flag_v2"]).sum())
    n_unsure = int((d["verdict"] == "u").sum())
    print(f"audited (train split): {n_flag_j} flagged claims, "
          f"{int((judged & ~d['flag_v2']).sum())} unflagged controls, {n_unsure} unsure (left out)\n")
    print(f"{'rule':10s} {'flags':>6s} {'answers flagged':>16s} {'audited':>8s} {'real':>5s}  precision [95% CI]")
    best, best_key, rates = None, None, {}
    for rule in RULES:
        f = flags[rule]
        keys = set(zip(d.loc[f, "id"], d.loc[f, "idx"].astype(int)))
        n_all, rate = len(keys), answer_rate(base, keys)
        m = f & judged
        n, x = int(m.sum()), int((m & (d["verdict"] == "r")).sum())
        if n:
            lo, hi = clopper_pearson(x, n)
            prec = f"{x / n:5.0%} [{lo:.0%}–{hi:.0%}]"
        else:
            lo, prec = 0.0, "  –"
        print(f"{rule:10s} {n_all:6d} {rate:16.1%} {n:8d} {x:5d}  {prec}")
        rates[rule] = rate
        if n >= 8:
            key = (lo, n_all)                 # rank by the CI lower bound, so a tiny lucky sample doesn't win
            if best_key is None or key > best_key:
                best, best_key = rule, key
    ctrl = judged & ~d["flag_v2"]
    if ctrl.any():
        print(f"\nunflagged controls judged a real disagreement: {int((ctrl & (d['verdict'] == 'r')).sum())} of "
              f"{int(ctrl.sum())} (a rough miss rate; v2 flags nothing here)")
    if best is None:
        print("\nNot enough judged items yet (need at least 8 flagged ones per rule). Run `python audit_labels.py`.")
        return None
    print(f"\nSuggested rule: {best} (highest lower bound on precision among rules with >= 8 audited flags).")
    if rates[best] < 0.05:
        print(f"Note: only {rates[best]:.1%} of answers would have a flagged claim; paste the table so we can weigh it.")
    print(f"Apply it with:  python audit_labels.py --apply {best}")
    return best


def apply_rule(base, d, cfg: dict, rule: str) -> None:
    import pandas as pd

    flags = rule_flags(d, cfg)
    keep = set(zip(d.loc[flags[rule], "id"], d.loc[flags[rule], "idx"].astype(int)))
    out = base.copy()
    kept = pd.Series([k in keep for k in zip(out["id"], out["idx"].astype(int))], index=out.index)
    drop = (out["label"] == "contradicted") & ~kept
    out.loc[drop, "label"] = "unverifiable"
    out["label_rule"] = rule
    out.to_parquet(CLAIMS, index=False)
    write_json_atomic(RULE_FILE, {"rule": rule, "applied_at": time.strftime("%Y-%m-%d %H:%M"),
                                  "dropped_v2_flags": int(drop.sum()), "audit_file": str(AUDIT.relative_to(ROOT))})
    print(f"rule '{rule}': {int(drop.sum())} of {int((base['label'] == 'contradicted').sum())} v2 flags dropped "
          f"-> {int((out['label'] == 'contradicted').sum())} contradicted claims remain")
    print("label shares:", {k: f"{v:.1%}" for k, v in out["label"].value_counts(normalize=True).items()})
    per = out.assign(_c=out["label"] == "contradicted").groupby(["split", "id"])["_c"].any().groupby("split").mean()
    print("answers with >=1 contradicted claim:", {k: f"{v:.1%}" for k, v in per.items()})
    n_cal_bad = int(out[out["split"] == "calibrate"].groupby("id")["label"].apply(lambda s: (s == "contradicted").any()).sum())
    if n_cal_bad < 15:
        print(f"WARNING: only {n_cal_bad} calibration answers keep a contradicted claim; the guarantee would be "
              "almost trivial. Paste this output before calibrating.")
    dev = out[out["split"] != "test"]          # decided from train + calibrate only
    rate = float(dev.assign(_c=dev["label"] == "contradicted").groupby("id")["_c"].any().mean())
    main_alpha = 0.10 if rate >= 0.15 else 0.05
    print(f"train+calibrate rate {rate:.1%} -> main alpha: {main_alpha}  (run: python calibrate.py --alpha {main_alpha})")
    print(f"wrote {CLAIMS.relative_to(ROOT)} (v2 backup: {BACKUP.relative_to(ROOT)}). "
          "If you re-run label_claims.py later, apply the rule again.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--apply", choices=RULES)
    ap.add_argument("--target", type=int, default=40, help="flagged items to aim for")
    ap.add_argument("--fake-nli", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    cfg = load_config()
    base, d, questions = load_all(cfg, args.fake_nli)
    if args.apply:
        apply_rule(base, d, cfg, args.apply)
    elif args.report:
        report(base, d, cfg)
    else:
        tr = d[d["split"] == "train"]
        print(f"{int(d['flag_v2'].sum())} v2-flagged claims in all splits; {int(tr['flag_v2'].sum())} in train "
              f"(+{int((~tr['flag_v2']).sum())} unflagged controls)")
        try:
            run_audit(d, questions, cfg, args.target)
        except (KeyboardInterrupt, EOFError):
            print(f"\nstopped; verdicts so far are saved in {AUDIT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
