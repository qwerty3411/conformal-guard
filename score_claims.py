"""Evidence score for every claim: retrieve MedlinePlus passages, then check support with MiniCheck.

    python score_claims.py --limit 10     # quick look
    python score_claims.py                # all claims in data/claims.parquet (resumable)

Needs: data/claims.parquet (label_claims.py) and data/evidence (build_index.py).
For each sentence claim:
  * retrieval: top-k MedlinePlus passages for "question + claim" (bge-small + FAISS)
  * evidence_support: MiniCheck-Flan-T5-Large probability that those passages support the claim
  * retrieval_sim: cosine similarity of the best passage (a weak relevance signal)
Writes cache/scores/<id>.json per question and data/scores.parquet (claims + scores).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

from cg_common import ROOT, load_config, write_json_atomic

MINICHECK_MODEL = "flan-t5-large"


def settings_key(cfg: dict) -> str:
    blob = {"evidence": cfg["evidence"], "minicheck": MINICHECK_MODEL, "v": 1}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]


class Checker:
    def __init__(self, fake: bool = False):
        self.fake = fake
        if not fake:
            from minicheck.minicheck import MiniCheck
            self.mc = MiniCheck(model_name=MINICHECK_MODEL, batch_size=16)

    def __call__(self, docs: list[str], claims: list[str]) -> list[float]:
        if self.fake:
            import re
            out = []
            for d, c in zip(docs, claims):
                cw = set(re.findall(r"[a-z]+", c.lower()))
                out.append(len(cw & set(re.findall(r"[a-z]+", d.lower()))) / max(1, len(cw)))
            return out
        _, probs, _, _ = self.mc.score(docs=docs, claims=claims)
        return [float(p) for p in probs]


def main() -> None:
    import pandas as pd

    from build_index import EvidenceIndex

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    cfg = load_config()
    key = settings_key(cfg)
    claims_path = ROOT / "data" / "claims.parquet"
    if not claims_path.exists():
        raise SystemExit("data/claims.parquet not found: run `python label_claims.py` first")
    if not (ROOT / cfg["evidence"]["dir"] / "passages.jsonl").exists():
        raise SystemExit("evidence index not found: run `python build_index.py` first")
    claims = pd.read_parquet(claims_path)
    questions = {json.loads(line)["id"]: json.loads(line) for line in
                 (ROOT / "data" / "questions_final.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    ids = list(dict.fromkeys(claims["id"]))
    if args.limit:
        ids = ids[:args.limit]
    out_dir = ROOT / "cache" / "scores"

    todo = []
    for qid in ids:
        f = out_dir / f"{qid}.json"
        if f.exists() and json.loads(f.read_text(encoding="utf-8")).get("settings_key") == key:
            continue
        todo.append(qid)
    print(f"{len(ids)} questions · {len(ids) - len(todo)} already scored · {len(todo)} to do")

    if todo:
        index = EvidenceIndex(cfg, fake=args.fake)
        print(f"evidence index: {len(index.passages)} passages ({'faiss' if index.faiss else 'numpy'})")
        checker = Checker(fake=args.fake)
        k = cfg["evidence"]["top_k"]
        t0 = time.monotonic()
        for n_done, qid in enumerate(todo, 1):
            rows = claims[claims["id"] == qid].sort_values("idx")
            q = questions[qid]["question"]
            hits = index.search([f"{q} {t}" for t in rows["text"]], k)
            docs = ["\n".join(p["text"] for _, p in h) for h in hits]
            support = checker(docs, list(rows["text"]))
            rec = {"id": qid, "settings_key": key, "scores": [
                {"idx": int(i), "evidence_support": round(s, 4), "retrieval_sim": round(h[0][0], 4),
                 "evidence_urls": [p["url"] for _, p in h]}
                for i, s, h in zip(rows["idx"], support, hits)]}
            write_json_atomic(out_dir / f"{qid}.json", rec)
            if args.limit:
                print(f"\n[{qid}] {q}")
                for (_, r), s, h in zip(rows.iterrows(), support, hits):
                    print(f"  {r['label']:13s} evidence={s:.2f} sim={h[0][0]:.2f} | {r['text'][:90]}")
                    print(f"      best passage: {h[0][1]['text'][:110]}")
            elif n_done % 50 == 0 or n_done == len(todo):
                el = time.monotonic() - t0
                print(f"{n_done}/{len(todo)} questions · {el / 60:.1f} min · "
                      f"ETA {(len(todo) - n_done) * el / n_done / 60:.0f} min", flush=True)

    # collect
    recs = []
    for qid in dict.fromkeys(claims["id"]):
        f = out_dir / f"{qid}.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("settings_key") == key:
                recs += [{"id": qid, **s} for s in r["scores"]]
    sc = pd.DataFrame(recs)
    if sc.empty:
        print("no scores yet")
        return
    sc["evidence_urls"] = sc["evidence_urls"].apply(json.dumps)
    df = claims.merge(sc, on=["id", "idx"], how="inner")
    dest = ROOT / "data" / "scores.parquet"
    df.to_parquet(dest, index=False)
    print(f"\nwrote {dest}: {len(df)} claims with scores from {df['id'].nunique()} answers")
    print("mean evidence support by label:", df.groupby("label")["evidence_support"].mean().round(3).to_dict())
    print("mean agreement by label:      ", df.groupby("label")["agree_mean"].mean().round(3).to_dict())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Finished questions are saved; run the same command again to resume.")
