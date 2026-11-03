"""Freeze the questions that have a complete set of answers and split them into train/calibrate/test.

    python resplit.py            # writes data/questions_final.jsonl (refuses to overwrite)
    python resplit.py --dry-run  # show the counts only

"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from pathlib import Path

from cg_common import load_config, path, settings_hash
from screen import screen


def cached(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_refused(logs: Path) -> set[str]:
    """Questions the content filter rejected (from refused_questions.jsonl or the error log)."""
    ids: set[str] = set()
    for name in ("refused_questions.jsonl", "generate_errors.jsonl"):
        f = logs / name
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            text = f"{r.get('body') or ''} {r.get('message') or ''}".lower()
            if name == "refused_questions.jsonl" or (r.get("status") == 400 and (
                    "content_policy" in text or "content management policy" in text)):
                ids.add(r["id"])
    return ids

FINAL = "data/questions_final.jsonl"
PROPORTIONS = {"train": 0.30, "calibrate": 0.40, "test": 0.30}
SEED = 20261027


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    h, n_samples = settings_hash(cfg), cfg["generation"]["samples_per_question"]
    cache_dir, logs = path(cfg, "cache"), path(cfg, "logs")
    refused = load_refused(logs)
    rows = [json.loads(line) for line in path(cfg, "questions").read_text(encoding="utf-8").splitlines() if line]

    keep, why = [], Counter()
    for r in rows:
        recs = [cached(cache_dir / r["id"] / f"{k}.json") for k in range(n_samples)]
        if not screen(r["question"])[0]:
            why["screened out"] += 1
        elif r["id"] in refused:
            why["rejected by content filter"] += 1
        elif not all(x and x.get("ok") and x.get("settings_hash") == h for x in recs):
            why["incomplete answers"] += 1
        else:
            keep.append(r)

    rng = random.Random(SEED)
    by_group: dict[str, list] = {}
    for r in keep:
        by_group.setdefault(r["group"], []).append(r)
    order = []
    for rs in by_group.values():          # stratify: interleave groups by within-group rank, then cut
        rng.shuffle(rs)
        order += [((i + rng.random()) / len(rs), r) for i, r in enumerate(rs)]
    order.sort(key=lambda t: t[0])
    n = len(order)
    n_train = round(n * PROPORTIONS["train"])
    n_cal = round(n * PROPORTIONS["calibrate"])
    out = []
    for i, (_, r) in enumerate(order):
        split = "train" if i < n_train else "calibrate" if i < n_train + n_cal else "test"
        out.append({**r, "orig_split": r["split"], "split": split})

    print(f"{len(rows)} questions · kept {n} · dropped {dict(why)}")
    print(f"{'group':12s} {'train':>7s} {'calibrate':>10s} {'test':>7s} {'total':>7s}")
    for gname in sorted(by_group):
        c = Counter(r["split"] for r in out if r["group"] == gname)
        print(f"{gname:12s} {c['train']:7d} {c['calibrate']:10d} {c['test']:7d} {sum(c.values()):7d}")
    c = Counter(r["split"] for r in out)
    print(f"{'all':12s} {c['train']:7d} {c['calibrate']:10d} {c['test']:7d} {n:7d}")
    if args.dry_run:
        return
    dest = path(cfg, "logs").parent / FINAL
    if dest.exists():
        raise SystemExit(f"{FINAL} already exists and is frozen; delete it only if no results have been computed yet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {dest} (frozen: commit it now)")


if __name__ == "__main__":
    main()
