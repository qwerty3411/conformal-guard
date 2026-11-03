"""Download MedLFQA at a pinned commit, pick 1,300 questions by group, and split them once.

    python prepare_data.py

Writes data/questions.jsonl: one row per question with id, group, source file, split,
question, must_have and nice_to_have reference statements. Run it once and commit the output:
the split must never change after you start looking at results.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request

from cg_common import load_config, path

RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/MedLFQA/{file}"


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", q.lower())).strip()


def fetch(cfg: dict, file: str) -> list[dict]:
    raw_dir = path(cfg, "raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    local = raw_dir / file
    if not local.exists():
        url = RAW_URL.format(repo=cfg["data"]["source_repo"], commit=cfg["data"]["source_commit"], file=file)
        print(f"downloading {file}")
        with urllib.request.urlopen(url, timeout=60) as r:
            local.write_bytes(r.read())
    return [json.loads(line) for line in local.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    cfg = load_config()
    d = cfg["data"]
    rng = random.Random(d["seed"])
    seen: set[str] = set()
    picked: dict[str, list[dict]] = {}

    for group, sources in d["groups"].items():
        rows: list[dict] = []
        for src in sources:
            cand = []
            for r in fetch(cfg, src["file"]):
                q = (r.get("Question") or "").strip()
                must = [s for s in (r.get("Must_have") or []) if s and s.strip()]
                key = norm(q)
                if not q or not must or key in seen:
                    continue
                seen.add(key)
                cand.append({"question": q, "must_have": must,
                             "nice_to_have": [s for s in (r.get("Nice_to_have") or []) if s and s.strip()],
                             "source": src["file"].replace("_test_MedLFQA.jsonl", "").replace("_test", "")})
            rng.shuffle(cand)
            take = len(cand) if src["take"] == "all" else int(src["take"])
            if take > len(cand):
                raise SystemExit(f"{src['file']}: wanted {take}, only {len(cand)} usable rows")
            rows += cand[:take]
        for r in rows:
            r["group"] = group
            r["id"] = f"{group}-{hashlib.sha1(r['question'].encode()).hexdigest()[:10]}"
        picked[group] = rows

    # stratified split with exact totals: interleave groups by within-group rank, then cut
    order = []
    for group, rows in picked.items():
        rng.shuffle(rows)
        n = len(rows)
        order += [((i + rng.random()) / n, r) for i, r in enumerate(rows)]
    order.sort(key=lambda t: t[0])
    sizes = d["splits"]
    total = sum(sizes.values())
    if total != len(order):
        raise SystemExit(f"splits add up to {total} but {len(order)} questions were picked")
    cuts, start = {}, 0
    for name, size in sizes.items():
        cuts[name] = (start, start + size)
        start += size
    out = []
    for idx, (_, r) in enumerate(order):
        r["split"] = next(name for name, (a, b) in cuts.items() if a <= idx < b)
        out.append(r)

    dest = path(cfg, "questions")
    with open(dest, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps({k: r[k] for k in ("id", "group", "source", "split", "question",
                                                   "must_have", "nice_to_have")}, ensure_ascii=False) + "\n")

    print(f"wrote {len(out)} questions to {dest}")
    print(f"{'group':12s} " + " ".join(f"{s:>10s}" for s in sizes) + "      total")
    for group in picked:
        counts = [sum(1 for r in out if r["group"] == group and r["split"] == s) for s in sizes]
        print(f"{group:12s} " + " ".join(f"{c:10d}" for c in counts) + f" {sum(counts):10d}")


if __name__ == "__main__":
    main()
