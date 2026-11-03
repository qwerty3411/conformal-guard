"""
    python screen.py            # list what the screen removes from data/questions.jsonl
"""

from __future__ import annotations

import json
import re
from collections import Counter

from cg_common import load_config, path

# Category -> whole-word patterns (case-insensitive). Deliberately narrow: ordinary medication,
# sexual-health and infection questions passed the filter in the 5,000-call run and stay in.
CATEGORIES: dict[str, list[str]] = {
    "illicit_drugs_or_overdose": [
        r"heroin", r"fentanyl", r"carfentanil", r"cocaine", r"crack cocaine", r"meth", r"methamphetamines?",
        r"mdma", r"ecstasy", r"lsd", r"ghb", r"psilocybin", r"magic mushrooms?", r"kratom",
        r"overdos(?:e|es|ed|ing)", r"lethal dose", r"fatal dose", r"get(?:ting)? high", r"recreational(?:ly)?",
    ],
    "dangerous_pathogens_or_toxins": [
        r"hantavirus(?:es)?", r"anthrax", r"ebola", r"marburg", r"smallpox", r"plague", r"botulinum",
        r"botulism", r"ricin", r"sarin", r"nerve agents?", r"bioweapons?", r"biological weapons?",
        r"tularemia", r"nipah", r"gain of function",
    ],
    "self_harm": [
        r"suicid(?:e|al)", r"self[- ]harm", r"kill (?:my|him|her)self", r"end (?:my|his|her) life",
    ],
}
_COMPILED = {c: re.compile(r"\b(?:" + "|".join(p) + r")\b", re.IGNORECASE) for c, p in CATEGORIES.items()}


def screen(question: str) -> tuple[bool, str | None, str | None]:
    """Return (allowed, category, matched text)."""
    for cat, rx in _COMPILED.items():
        m = rx.search(question)
        if m:
            return False, cat, m.group(0)
    return True, None, None


OUT_OF_SCOPE_REPLY = ("This question is outside what this assistant is calibrated for "
                      "(illicit drugs, overdoses, dangerous pathogens or self-harm). "
                      "If this is an emergency, contact local emergency services or a poison control centre.")


def main() -> None:
    cfg = load_config()
    rows = [json.loads(line) for line in path(cfg, "questions").read_text(encoding="utf-8").splitlines() if line]
    removed = []
    for r in rows:
        ok, cat, term = screen(r["question"])
        if not ok:
            removed.append((r, cat, term))
    by_cat = Counter(c for _, c, _ in removed)
    by_split = Counter(r["split"] for r, _, _ in removed)
    print(f"{len(removed)} of {len(rows)} questions screened out · by category {dict(by_cat)} · by split {dict(by_split)}\n")
    for r, cat, term in removed:
        print(f"{r['id']:24s} {r['split']:9s} {cat:30s} [{term}] {r['question'][:110]}")


if __name__ == "__main__":
    main()
