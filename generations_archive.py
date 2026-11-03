"""Pack the cached Muse Spark generations of the frozen question set into one file, or unpack it.

    python generations_archive.py --export     # cache/generations -> data/generations.jsonl.gz (commit this)
    python generations_archive.py --restore    # data/generations.jsonl.gz -> cache/generations (after cloning)
"""

from __future__ import annotations

import argparse
import gzip
import json

from cg_common import ROOT, load_config, path, write_json_atomic

ARCHIVE = ROOT / "data" / "generations.jsonl.gz"


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--export", action="store_true")
    g.add_argument("--restore", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    cache = path(cfg, "cache")
    n = cfg["generation"]["samples_per_question"]
    if args.export:
        qs = [json.loads(l) for l in (ROOT / "data" / "questions_final.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()]
        missing, count = [], 0
        with gzip.open(ARCHIVE, "wt", encoding="utf-8") as f:
            for q in qs:
                for k in range(n):
                    p = cache / q["id"] / f"{k}.json"
                    if not p.exists():
                        missing.append(f"{q['id']}/{k}")
                        continue
                    f.write(json.dumps(json.loads(p.read_text(encoding="utf-8")), ensure_ascii=False) + "\n")
                    count += 1
        print(f"wrote {ARCHIVE.relative_to(ROOT)}: {count} samples for {len(qs)} questions "
              f"({ARCHIVE.stat().st_size / 1e6:.1f} MB)")
        if missing:
            print(f"WARNING: {len(missing)} samples missing, e.g. {missing[:3]}")
    else:
        if not ARCHIVE.exists():
            raise SystemExit(f"{ARCHIVE.relative_to(ROOT)} not found")
        count = 0
        with gzip.open(ARCHIVE, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    write_json_atomic(cache / r["id"] / f"{r['k']}.json", r)
                    count += 1
        print(f"restored {count} samples into {cache.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
