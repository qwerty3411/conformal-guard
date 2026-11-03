"""Conformal Guard: score each sentence of an answer, keep the ones above the calibrated threshold, decide.

    python guard.py --id consumer-1ea4af4628          # replay one cached answer, sentence by sentence
    python guard.py --question "Is uveitis an autoimmune disease?"
    python guard.py --id consumer-1ea4af4628 --alpha 0.02
    python guard.py --firewall --id consumer-1ea4af4628   # the same decision through LlamaFirewall
    python guard.py --check 25                        # does guard reproduce calibrate.py's scores?
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
from pathlib import Path

import numpy as np

from cg_common import ROOT, load_config

THRESHOLDS = ROOT / "results" / "thresholds.json"
TEST_CLAIMS = ROOT / "results" / "test_claims.parquet"
DOMAIN_QUANTILE = 0.01          # questions less similar to the calibration set than 99% of calibration questions are
REFERENCE = "the MedLFQA reference answer"


# ----------------------------------------------------------------------------- pure functions (tested)
def load_thresholds(p: Path = THRESHOLDS) -> dict:
    if not p.exists():
        raise SystemExit("results/thresholds.json not found: run `python calibrate.py` first")
    th = json.loads(p.read_text(encoding="utf-8"))
    th["tau"] = {float(a): float(t) for a, t in th["tau"].items()}
    return th


def combine(features: dict[str, list[float]], th: dict) -> np.ndarray:
    """The calibrated logistic-regression combiner: P(sentence is not false), as calibrate.py fitted it."""
    X = np.column_stack([np.asarray(features[f], dtype=float) for f in th["features"]])
    z = (X - np.asarray(th["scaler_mean"])) / np.asarray(th["scaler_scale"])
    logit = z @ np.asarray(th["coef"]) + th["intercept"]
    return 1.0 / (1.0 + np.exp(-logit))


def nearest_alpha(th: dict, alpha: float) -> float:
    return min(th["tau"], key=lambda a: abs(a - alpha))


def decide(combined: np.ndarray, th: dict, alpha: float | None = None) -> dict:
    """Filter at `alpha` and make the ship / review / block decision at the main alpha (as calibrated)."""
    main = th.get("main_alpha", 0.10)
    alpha = nearest_alpha(th, main if alpha is None else alpha)
    kept = combined > th["tau"][alpha]
    kept_main = combined > th["tau"][nearest_alpha(th, main)]
    weakest = float(combined.min()) if len(combined) else 1.0
    if weakest >= th["ship_lambda"]:
        decision = "ship"
    elif kept_main.any():
        decision = "review"
    else:
        decision = "block"
    return {"alpha": alpha, "tau": th["tau"][alpha], "kept": kept, "decision": decision,
            "weakest": weakest, "ship_lambda": th["ship_lambda"], "main_alpha": main}


def _pct(x: float) -> str:
    """0.05 -> '5%', 0.925 -> '92.5%' (no rounding surprises from :.0%)."""
    return f"{round(100 * x, 2):g}%"


def review_coverage(th: dict, alpha: float, p: Path = TEST_CLAIMS) -> float | None:
    """Share of held-out test answers sent to review whose filtered version has no contradicted sentence.

    The conformal bound is an average over all answers. Answers sent to review are selected because they
    look risky, so the bound is not guaranteed for them; this is what was observed for them instead.
    """
    try:
        import pandas as pd
        t = pd.read_parquet(p, columns=["id", "is_false", "combined"])
    except Exception:  # noqa: BLE001
        return None
    tau_main = th["tau"][nearest_alpha(th, th.get("main_alpha", alpha))]
    tau_a = th["tau"][nearest_alpha(th, alpha)]
    g = t.groupby("id")["combined"]
    reviewed = (g.min() < th["ship_lambda"]) & (g.max() > tau_main)       # same rule as decide()
    ids = reviewed[reviewed].index
    if len(ids) == 0:
        return None
    wrong_kept = (t["is_false"] & (t["combined"] > tau_a)).groupby(t["id"]).any()
    return float(1.0 - wrong_kept.reindex(ids).mean())


def certificate(th: dict, alpha: float, n_kept: int, n_total: int, decision: str,
                summary: dict | None = None, review_obs: float | None = None) -> str:
    rule = th.get("label_rule", "v2")
    n = th.get("n_calibration_answers", "?")
    main = th.get("main_alpha", alpha)
    src = (f"Calibrated on {n} other MedLFQA questions answered by Muse Spark; 'contradicts' means an NLI "
           f"label pipeline (rule '{rule}') finds a conflict with {REFERENCE}. This is a guarantee about the "
           "method, averaged over questions like these, not a promise about this one answer.")
    if decision == "ship":
        ab = (summary or {}).get("abstention", {})
        seen, given = ab.get("p_shipped_and_wrong"), ab.get("p_wrong_given_shipped")
        obs = ""
        if seen is not None:
            obs = f" (held-out test: {seen:.1%} of all answers" + (
                f"; {given:.1%} of the answers that shipped)" if given is not None else ")")
        return (f"Shipped unchanged. The ship rule is calibrated so that at most {_pct(main)} of all answers are "
                f"shipped unchanged while containing a sentence that contradicts {REFERENCE}{obs}. {src}")
    if decision == "review":
        obs = f"; on held-out answers sent to review it was {review_obs:.1%}" if review_obs is not None else ""
        return (f"Filtered to {n_kept} of {n_total} sentences and sent to a person for review. Across all "
                f"answers, filtering at alpha = {alpha:g} leaves at least {_pct(1 - alpha)} with no sentence that "
                f"contradicts {REFERENCE}. That average is not guaranteed for answers sent to review, because "
                f"the system selected them{obs}. {src}")
    if decision == "block":
        return f"Blocked: no sentence cleared the calibrated threshold at alpha = {alpha:g}. {src}"
    if decision == "unscored":
        return ("Not scored: no resamples of this question were available, so the sentences could not be scored "
                "and no guarantee is claimed. The full answer goes to a person.")
    return (f"Not covered: this question is less similar to the calibration questions than "
            f"{_pct(1 - DOMAIN_QUANTILE)} of them are to each other, so no guarantee is claimed. "
            "The full answer goes to a person.")


# ----------------------------------------------------------------------------- resamples
class CachedSampler:
    """Replay: the 5 cached Muse Spark resamples for a question from the frozen question set."""

    def __init__(self, cfg: dict):
        from cg_common import path
        self.gen_dir = path(cfg, "cache")
        self.n = cfg["nli"].get("agreement_resamples", 5)
        self.by_text, self.by_id = {}, {}
        for line in (ROOT / "data" / "questions_final.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                q = json.loads(line)
                self.by_text[q["question"].strip()] = q
                self.by_id[q["id"]] = q

    def find(self, question: str) -> dict | None:
        return self.by_text.get(question.strip())

    def answer(self, qid: str) -> str:
        return json.loads((self.gen_dir / qid / "0.json").read_text(encoding="utf-8"))["content"]

    def __call__(self, question: str) -> list[str] | None:
        q = self.find(question)
        if q is None:
            return None
        return [json.loads((self.gen_dir / q["id"] / f"{k}.json").read_text(encoding="utf-8"))["content"]
                for k in range(1, 1 + self.n)]


class MetaSampler:
    """Live path: draw resamples from the Meta Model API with the frozen generation settings.

    Not used in this submission (the account was restricted after Oct 26); enabled only with CG_LIVE=1.
    """

    def __init__(self, cfg: dict):
        if os.getenv("CG_LIVE") != "1":
            raise RuntimeError("live sampling is off; set CG_LIVE=1 to call the Meta Model API")
        from cg_common import make_client, messages_for
        self.cfg, self.client, self.messages_for = cfg, make_client(cfg), messages_for
        self.n = cfg["nli"].get("agreement_resamples", 5)

    async def _one(self, question: str) -> str:
        g = self.cfg["generation"]
        r = await self.client.chat.completions.create(
            model=self.cfg["api"]["model"], messages=self.messages_for(self.cfg, question),
            temperature=g["temperature"], reasoning_effort=g["reasoning_effort"],
            max_completion_tokens=g["max_completion_tokens"])
        return r.choices[0].message.content or ""

    def __call__(self, question: str) -> list[str]:
        async def run():
            return await asyncio.gather(*[self._one(question) for _ in range(self.n)])
        return list(asyncio.run(run()))


# ----------------------------------------------------------------------------- the guard
class ConformalGuard:
    def __init__(self, cfg: dict | None = None, th: dict | None = None, fake: bool = False):
        from build_index import EvidenceIndex
        from label_claims import NLI
        from score_claims import Checker

        self.cfg = cfg or load_config()
        self.th = th or load_thresholds()
        sp = ROOT / "results" / "summary.json"
        self.summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else None
        n = self.cfg["nli"]
        self.nli = NLI(n["model"], n["batch_size"], n["max_length"], fake=fake)
        self.index = EvidenceIndex(self.cfg, fake=fake)
        self.checker = Checker(fake=fake)
        self._review_obs: dict[float, float | None] = {}
        self._init_domain()

    def _init_domain(self) -> None:
        """Nearest-neighbour similarity of each calibration question to the others sets the in-domain bar."""
        cal = [json.loads(l)["question"] for l in
               (ROOT / "data" / "questions_final.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip() and json.loads(l)["split"] == "calibrate"]
        self.cal_emb = self.index.embedder(cal)
        sims = self.cal_emb @ self.cal_emb.T
        np.fill_diagonal(sims, -1.0)
        self.domain_bar = float(np.quantile(sims.max(axis=1), DOMAIN_QUANTILE))

    def domain(self, question: str) -> tuple[bool, float]:
        s = float((self.index.embedder([question]) @ self.cal_emb.T).max())
        return s >= self.domain_bar, s

    def score(self, question: str, answer: str, resamples: list[str]) -> list[dict]:
        """Sentence claims with the same features and rounding as label_claims.py / score_claims.py."""
        from label_claims import split_sentences

        n = self.cfg["nli"]
        sents = split_sentences(answer, self.cfg["claims"]["min_words"])
        if not sents:
            return []
        rs = resamples[:n.get("agreement_resamples", len(resamples))]
        probs = self.nli([(r, s) for s, _, _ in sents for r in rs])
        hits = self.index.search([f"{question} {s}" for s, _, _ in sents], self.cfg["evidence"]["top_k"])
        support = self.checker(["\n".join(p["text"] for _, p in h) for h in hits], [s for s, _, _ in sents])
        claims = []
        for i, ((s, a, b), h, sup) in enumerate(zip(sents, hits, support)):
            ap = probs[i * len(rs):(i + 1) * len(rs)]
            claims.append({
                "idx": i, "text": s, "start": a, "end": b, "n_words": len(s.split()),
                "agree_mean": round(statistics.mean(p[0] for p in ap), 4) if ap else 0.0,
                "agree_frac": round(statistics.mean(p[0] >= n["entail_threshold"] for p in ap), 4) if ap else 0.0,
                "evidence_support": round(float(sup), 4), "retrieval_sim": round(h[0][0], 4),
                "evidence": [{"title": p["title"], "url": p["url"], "text": p["text"]} for _, p in h]})
        feats = {"agree_mean": [c["agree_mean"] for c in claims], "agree_frac": [c["agree_frac"] for c in claims],
                 "evidence_support": [c["evidence_support"] for c in claims],
                 "retrieval_sim": [c["retrieval_sim"] for c in claims],
                 "log_words": [math.log(max(c["n_words"], 1)) for c in claims],
                 "position": [c["idx"] / len(claims) for c in claims]}
        for c, p in zip(claims, combine(feats, self.th)):
            c["combined"] = float(p)
        return claims

    def review_obs(self, alpha: float) -> float | None:
        if alpha not in self._review_obs:
            self._review_obs[alpha] = review_coverage(self.th, alpha)
        return self._review_obs[alpha]

    def run(self, question: str, answer: str, resamples: list[str] | None, alpha: float | None = None) -> dict:
        in_domain, sim = self.domain(question)
        unscored = not resamples
        if unscored:                                   # cannot score agreement: a person has to look
            a = nearest_alpha(self.th, alpha or self.th["main_alpha"])
            claims, d = [], {"alpha": a, "tau": self.th["tau"][a], "decision": "review", "kept": np.array([], bool)}
        else:
            claims = self.score(question, answer, resamples)
            d = decide(np.array([c["combined"] for c in claims]), self.th, alpha)
        for c, k in zip(claims, d["kept"]):
            c["kept"] = bool(k)
        decision = d["decision"] if in_domain else "out_of_domain"
        kept_text = " ".join(c["text"] for c in claims if c.get("kept"))
        delivered = {"ship": answer, "review": kept_text, "block": "", "out_of_domain": ""}[decision]
        if unscored and decision == "review":
            delivered = ""                             # nothing was filtered: the full answer goes to a person
        cert_kind = "unscored" if unscored and decision == "review" else decision
        cert = certificate(self.th, d["alpha"], sum(c.get("kept", False) for c in claims), len(claims), cert_kind,
                           self.summary, self.review_obs(d["alpha"]) if cert_kind == "review" else None)
        return {"question": question, "answer": answer, "claims": claims, "alpha": d["alpha"],
                "tau": d.get("tau"), "ship_lambda": self.th["ship_lambda"], "decision": decision,
                "delivered": delivered, "filtered_answer": kept_text,
                "in_domain": in_domain, "domain_similarity": round(sim, 4), "domain_bar": round(self.domain_bar, 4),
                "certificate": cert}


_GUARD: ConformalGuard | None = None
_LAST: dict | None = None


def get_guard() -> ConformalGuard:
    """One shared guard: LlamaFirewall creates a new scanner object for every scan, so models live here."""
    global _GUARD
    if _GUARD is None:
        _GUARD = ConformalGuard(fake=os.getenv("CG_FAKE") == "1")
    return _GUARD


def last_result() -> dict | None:
    """The full result of the most recent scan (LlamaFirewall itself only passes on BLOCK/REVIEW reasons)."""
    return _LAST


# ----------------------------------------------------------------------------- LlamaFirewall scanner
def register_scanner() -> str:
    from llamafirewall import Role, ScanDecision, Scanner, ScanResult, register_llamafirewall_scanner

    @register_llamafirewall_scanner("conformal_guard")
    class ConformalGuardScanner(Scanner):
        """Assistant-message scanner: ALLOW = ship, HUMAN_IN_THE_LOOP_REQUIRED = review / out of domain,
        BLOCK = nothing survives. The question is the latest user message in the trace."""

        def __init__(self) -> None:
            super().__init__(scanner_name="conformal_guard")

        async def scan(self, message, past_trace=None) -> ScanResult:
            global _LAST
            question = next((m.content for m in reversed(past_trace or []) if m.role == Role.USER), None)
            if question is None:
                return ScanResult(decision=ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED, score=1.0,
                                  reason="conformal_guard: no user question in the trace, cannot assess")
            guard = get_guard()
            sampler = MetaSampler(guard.cfg) if os.getenv("CG_LIVE") == "1" else CachedSampler(guard.cfg)
            resamples = await asyncio.to_thread(sampler, question)
            res = await asyncio.to_thread(guard.run, question, message.content, resamples)
            _LAST = res
            decision = {"ship": ScanDecision.ALLOW, "review": ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED,
                        "block": ScanDecision.BLOCK,
                        "out_of_domain": ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED}[res["decision"]]
            n = len(res["claims"])
            dropped = 0.0 if res["decision"] == "ship" or not n else sum(not c["kept"] for c in res["claims"]) / n
            return ScanResult(decision=decision, score=round(dropped, 4),      # share of sentences filtered out
                              reason=f"conformal_guard: {res['decision']}. {res['certificate']}")

    return "conformal_guard"


def firewall_scan(question: str, answer: str):
    from llamafirewall import AssistantMessage, LlamaFirewall, Role, UserMessage

    name = register_scanner()
    fw = LlamaFirewall(scanners={Role.ASSISTANT: [name]})
    return fw.scan(AssistantMessage(content=answer), trace=[UserMessage(content=question)])


# ----------------------------------------------------------------------------- CLI
def show(res: dict, labels: dict | None = None) -> None:
    print(f"\nQUESTION: {res['question']}")
    print(f"domain similarity {res['domain_similarity']:.3f} (bar {res['domain_bar']:.3f}) -> "
          f"{'in domain' if res['in_domain'] else 'OUT OF DOMAIN'}")
    weakest = min((c["combined"] for c in res["claims"]), default=float("nan"))
    print(f"\nDECISION: {res['decision'].upper()}   (weakest sentence {weakest:.4f}; "
          f"ship threshold {res['ship_lambda']:.4f})")
    shipped = res["decision"] == "ship"
    print(f"\nSentence filter at alpha {res['alpha']:g}, tau {res['tau']:.4f}"
          + ("  (not applied: the answer was shipped unchanged)" if shipped else ""))
    for c in res["claims"]:
        lab = f" [{labels.get(c['idx'])}]" if labels else ""
        mark = "sent" if shipped else ("KEEP" if c["kept"] else "drop")
        print(f"  {mark}  {c['combined']:.4f}{lab}  {c['text']}")
    print(f"\nDELIVERED: {res['delivered'] or '(nothing delivered automatically)'}")
    print(f"CERTIFICATE: {res['certificate']}")


def check(n: int, fake: bool) -> None:
    import pandas as pd

    guard = ConformalGuard(fake=fake)
    sampler = CachedSampler(guard.cfg)
    test = pd.read_parquet(ROOT / "results" / "test_claims.parquet")
    ids = list(dict.fromkeys(test["id"]))[:n]
    diffs, same_kept, total = [], 0, 0
    for qid in ids:
        q = sampler.by_id[qid]
        res = guard.run(q["question"], sampler.answer(qid), sampler(q["question"]))
        ref = test[test["id"] == qid].sort_values("idx")
        tau = guard.th["tau"][nearest_alpha(guard.th, guard.th["main_alpha"])]
        for c, (_, r) in zip(res["claims"], ref.iterrows()):
            diffs.append(abs(c["combined"] - r["combined"]))
            same_kept += (c["combined"] > tau) == (r["combined"] > tau)
            total += 1
    print(f"{len(ids)} test answers, {total} sentences: max |combined difference| {max(diffs):.5f}, "
          f"mean {np.mean(diffs):.5f}; same keep/drop decision for {same_kept}/{total}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id")
    ap.add_argument("--question")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--firewall", action="store_true", help="run the decision through LlamaFirewall")
    ap.add_argument("--check", type=int, metavar="N", help="compare with results/test_claims.parquet")
    ap.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.fake:
        os.environ["CG_FAKE"] = "1"
    if args.check:
        return check(args.check, args.fake)
    cfg = load_config()
    sampler = CachedSampler(cfg)
    q = sampler.by_id.get(args.id) if args.id else sampler.find(args.question or "")
    if q is None:
        raise SystemExit("question not found in data/questions_final.jsonl (replay needs a cached answer)")
    answer = sampler.answer(q["id"])
    labels = None
    try:
        import pandas as pd
        lab = pd.read_parquet(ROOT / "data" / "claims.parquet")
        lab = lab[lab["id"] == q["id"]]
        labels = dict(zip(lab["idx"].astype(int), lab["label"]))
    except Exception:  # noqa: BLE001
        pass
    print(f"{q['id']} · split {q['split']} · group {q['group']}")
    if args.firewall:
        r = firewall_scan(q["question"], answer)
        print(f"LlamaFirewall decision: {r.decision.value} (score {r.score})\nreason: {r.reason}")
        if last_result():
            show(last_result(), labels)
        return
    res = get_guard().run(q["question"], answer, sampler(q["question"]), args.alpha)
    show(res, labels)


if __name__ == "__main__":
    main()
