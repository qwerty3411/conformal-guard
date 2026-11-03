"""Turn each cached Muse Spark answer into sentence claims, label them, and score agreement. No API calls.

    python label_claims.py --limit 10    # quick look: labels and agreement for 10 answers
    python label_claims.py               # all questions in data/questions_final.jsonl (resumable)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

from cg_common import ROOT, load_config, path, write_json_atomic

ABBREVIATIONS = {"e.g.", "i.e.", "etc.", "vs.", "dr.", "mr.", "mrs.", "ms.", "approx.", "no.", "st.",
                 "u.s.", "u.k.", "fig.", "al.", "mg.", "ml.", "min.", "max.", "inc.", "jr.", "sr."}
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str, min_words: int = 4) -> list[tuple[str, int, int]]:
    """Deterministic sentence splitter. Returns (sentence, start, end) character spans."""
    text = text.strip()
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        candidate = text[start:m.start()].strip()
        last = candidate.split()[-1].lower().lstrip("(\"'[") if candidate.split() else ""
        if last in ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last):
            continue                       # "e.g. Advil" or "A. fumigatus": not a boundary
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    out = []
    for a, b in spans:
        s = text[a:b].strip()
        if len(s.split()) >= min_words:
            a2 = text.index(s, a)
            out.append((s, a2, a2 + len(s)))
    return out


class NLI:
    """Batched entailment/contradiction probabilities from a local MNLI-style model."""

    def __init__(self, model_name: str, batch_size: int, max_length: int, fake: bool = False):
        self.fake, self.batch_size, self.max_length = fake, batch_size, max_length
        if fake:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.i_ent = next(i for name, i in labels.items() if name.startswith("entail"))
        self.i_con = next(i for name, i in labels.items() if name.startswith("contra"))
        print(f"NLI model {model_name} on {self.device}")

    def __call__(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
        if self.fake:  # deterministic stand-in for tests without downloading the model
            out = []
            for p, h in pairs:
                hp = set(re.findall(r"[a-z]+", h.lower()))
                ov = len(hp & set(re.findall(r"[a-z]+", p.lower()))) / max(1, len(hp))
                out.append((ov, 0.9 if " not " in f" {p.lower()} " and ov > 0.5 else 0.05))
            return out
        res: list[tuple[float, float]] = [None] * len(pairs)  # type: ignore[list-item]
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation="only_first",
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.model(**enc).logits.float().softmax(-1).cpu().tolist()
            for i, p in zip(idx, probs):
                res[i] = (p[self.i_ent], p[self.i_con])
        return res


def label_question(q: dict, answers: list[str], nli: NLI, cfg: dict) -> dict:
    c, n = cfg["claims"], cfg["nli"]
    sents = split_sentences(answers[0], c["min_words"])
    refs = [s for s in q["must_have"] + q["nice_to_have"] if s.strip()]
    joined = " ".join(refs)
    resamples = answers[1:]
    pairs: list[tuple[str, str]] = []
    for s, _, _ in sents:
        pairs += [(r, s) for r in refs] + [(joined, s)] + [(r, s) for r in resamples]
    probs = nli(pairs) if pairs else []
    claims, pos, width = [], 0, len(refs) + 1 + len(resamples)
    for i, (s, a, b) in enumerate(sents):
        block = probs[pos:pos + width]
        pos += width
        ref_part, agree_part = block[:len(refs) + 1], block[len(refs) + 1:]
        ent = max(p[0] for p in ref_part)
        con = max(p[1] for p in ref_part[:len(refs)]) if refs else 0.0
        if con >= n["contradict_threshold"] and con > ent:
            label = "contradicted"
        elif ent >= n["entail_threshold"]:
            label = "supported"
        else:
            label = "unverifiable"
        claims.append({"idx": i, "text": s, "start": a, "end": b, "n_words": len(s.split()),
                       "ref_entail": round(ent, 4), "ref_contradict": round(con, 4), "label": label,
                       "agree_mean": round(statistics.mean(p[0] for p in agree_part), 4) if agree_part else None,
                       "agree_frac": round(statistics.mean(p[0] >= n["entail_threshold"] for p in agree_part), 4)
                       if agree_part else None})
    return {"id": q["id"], "split": q["split"], "group": q["group"], "n_refs": len(refs), "claims": claims}


def settings_key(cfg: dict) -> str:
    return hashlib.sha256(json.dumps({"claims": cfg["claims"], "nli": cfg["nli"]}, sort_keys=True).encode()).hexdigest()[:12]


def collect(cfg: dict, questions: list[dict], key: str) -> None:
    import pandas as pd

    rows, per_answer = [], []
    for q in questions:
        rec = json.loads((ROOT / "cache" / "claims" / f"{q['id']}.json").read_text(encoding="utf-8"))
        if rec.get("settings_key") != key:
            continue
        labels = [c["label"] for c in rec["claims"]]
        per_answer.append((q["split"], len(labels), "contradicted" in labels,
                           any(l != "supported" for l in labels)))
        for c in rec["claims"]:
            rows.append({"id": q["id"], "split": q["split"], "group": q["group"], **c})
    df = pd.DataFrame(rows)
    dest = ROOT / "data" / "claims.parquet"
    df.to_parquet(dest, index=False)
    n_ans = len(per_answer)
    print(f"\nwrote {dest}: {len(df)} claims from {n_ans} answers")
    print(f"claims per answer: mean {statistics.mean(p[1] for p in per_answer):.1f}, "
          f"median {statistics.median(p[1] for p in per_answer):.0f}")
    print("label shares:", {k: f"{v:.1%}" for k, v in df["label"].value_counts(normalize=True).items()})
    mu_c = statistics.mean(p[2] for p in per_answer)
    mu_s = statistics.mean(p[3] for p in per_answer)
    print(f"answers with >=1 CONTRADICTED claim (contradiction definition): {mu_c:.1%}")
    print(f"answers with >=1 claim NOT SUPPORTED (strict definition):       {mu_s:.1%}")
    print("Rule of thumb: the definition whose rate is well above alpha but not near 100% will keep the most claims.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fake-nli", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    cfg = load_config()
    key = settings_key(cfg)
    final = ROOT / "data" / "questions_final.jsonl"
    if not final.exists():
        raise SystemExit("data/questions_final.jsonl not found: run `python resplit.py` first")
    questions = [json.loads(line) for line in final.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]
    gen_dir, out_dir = path(cfg, "cache"), ROOT / "cache" / "claims"
    n_samples = cfg["generation"]["samples_per_question"]

    todo = []
    for q in questions:
        f = out_dir / f"{q['id']}.json"
        if f.exists() and json.loads(f.read_text(encoding="utf-8")).get("settings_key") == key:
            continue
        todo.append(q)
    print(f"{len(questions)} questions · {len(questions) - len(todo)} already labeled · {len(todo)} to do")
    if todo:
        nli = NLI(cfg["nli"]["model"], cfg["nli"]["batch_size"], cfg["nli"]["max_length"], fake=args.fake_nli)
        t0 = time.monotonic()
        for i, q in enumerate(todo, 1):
            answers = [json.loads((gen_dir / q["id"] / f"{k}.json").read_text(encoding="utf-8"))["content"]
                       for k in range(n_samples)]
            rec = label_question(q, answers, nli, cfg)
            rec["settings_key"] = key
            write_json_atomic(out_dir / f"{q['id']}.json", rec)
            if args.limit:
                print(f"\n[{q['id']}] {q['question']}")
                for c in rec["claims"]:
                    print(f"  {c['label']:13s} agree={c['agree_mean']:.2f} ref={c['ref_entail']:.2f}/"
                          f"{c['ref_contradict']:.2f} | {c['text'][:110]}")
            elif i % 25 == 0 or i == len(todo):
                el = time.monotonic() - t0
                print(f"{i}/{len(todo)} questions · {el / 60:.1f} min · ETA {(len(todo) - i) * el / i / 60:.0f} min",
                      flush=True)
    collect(cfg, questions, key)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Finished questions are saved; run the same command again to resume.")
