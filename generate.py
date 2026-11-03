"""Generate the answer + resamples for every question, cached to disk and resumable.

    python generate.py --limit 20        # pilot: first 20 questions, then read the outputs
    python generate.py                   # full run (all splits); safe to stop and restart
    python generate.py --dry-run         # show what would run and the projected cost
    python generate.py --retry-failed    # redo samples that were cut off or came back empty

Each sample saved as cache/generations/<question id>/<k>.json (k = 0 is the answer shown to
the user, 1..5 are resamples) together with the settings hash, token usage and cost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from cg_common import (RateLimiter, call_with_retry, cost_usd, error_info, fmt_money, load_config,
                       make_client, messages_for, path, settings_hash, usage_dict, write_json_atomic)
from screen import screen

ACCOUNT_WORDS = ("credit", "balance", "quota", "billing", "insufficient", "payment", "spend",
                 "limit exceeded", "restricted", "blocked")


def load_questions(cfg: dict, split: str, limit: int | None) -> list[dict]:
    qpath = path(cfg, "questions")
    if not qpath.exists():
        raise SystemExit("data/questions.jsonl not found: run `python prepare_data.py` first")
    rows = [json.loads(line) for line in qpath.read_text(encoding="utf-8").splitlines() if line.strip()]
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    return rows[:limit] if limit else rows


def cached(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def is_content_flag(info: dict) -> bool:
    text = f"{info.get('body') or ''} {info.get('message') or ''}".lower()
    return info.get("status") == 400 and ("content_policy" in text or "content management policy" in text)


def load_refused(logs: Path) -> set[str]:
    """Questions the content filter has rejected before (from either log)."""
    ids: set[str] = set()
    ref = logs / "refused_questions.jsonl"
    if ref.exists():
        ids |= {json.loads(line)["id"] for line in ref.read_text(encoding="utf-8").splitlines() if line.strip()}
    err = logs / "generate_errors.jsonl"
    if err.exists():
        for line in err.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if is_content_flag(r):
                    ids.add(r["id"])
    return ids


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="all", choices=["all", "train", "calibrate", "test"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    g, model, h = cfg["generation"], cfg["api"]["model"], settings_hash(cfg)
    budget = cfg["budgets_usd"]["generate"]
    max_flags = cfg.get("safety", {}).get("stop_after_flags", 1)
    cache_dir = path(cfg, "cache")
    logs = path(cfg, "logs")
    logs.mkdir(parents=True, exist_ok=True)
    questions = load_questions(cfg, args.split, args.limit)
    refused = load_refused(logs)
    screened = [q for q in questions if not screen(q["question"])[0]]
    sendable = {q["id"] for q in questions if screen(q["question"])[0] and q["id"] not in refused}

    # inventory the cache
    todo, spent, done, other_hash, failed = [], 0.0, 0, 0, 0
    for q in questions:
        for k in range(g["samples_per_question"]):
            rec = cached(cache_dir / q["id"] / f"{k}.json")
            if rec is not None and rec.get("settings_hash") != h:
                other_hash += 1
                continue
            if rec is not None:
                spent += rec.get("cost_usd", 0)
            if q["id"] not in sendable:
                continue
            if rec is None:
                todo.append((q, k))
            elif rec.get("ok"):
                done += 1
            elif args.retry_failed:
                todo.append((q, k))
            else:
                failed += 1
    if other_hash:
        raise SystemExit(f"{other_hash} cached samples were made with different settings (hash != {h}). "
                         "Move cache/generations aside or restore config.yaml; never mix the two.")
    n_refused = len([q for q in questions if q["id"] in refused])
    avg = spent / max(1, done)
    print(f"{len(questions)} questions · {len(screened)} screened out · {n_refused} refused earlier · "
          f"{done} samples done · {failed} failed · {len(todo)} to run · spent so far {fmt_money(cfg, spent)} · "
          f"budget {fmt_money(cfg, budget)}")
    if done:
        print(f"projected cost of remaining samples: {fmt_money(cfg, avg * len(todo))}")
    if args.dry_run or not todo:
        return

    client = make_client(cfg)
    limiter = RateLimiter(cfg["api"]["rpm_limit"])
    sem_answers = asyncio.Semaphore(cfg.get("safety", {}).get("answer_phase_concurrency", 8))  # few in flight, so a stop is quick
    sem_resamples = asyncio.Semaphore(cfg["api"]["max_concurrency"])
    state = {"spent": spent, "new": 0, "ok": 0, "bad": 0, "errors": 0, "flags": 0, "stopped": False,
             "budget_hit": False, "streak": 0, "stop_reason": None, "seen": set(), "refused": set(refused)}
    t_start = time.monotonic()
    err_log = open(logs / "generate_errors.jsonl", "a", encoding="utf-8")
    ref_log = open(logs / "refused_questions.jsonl", "a", encoding="utf-8")

    def stop(reason: str) -> None:
        if not state["stopped"]:
            state["stopped"] = True
            state["stop_reason"] = reason

    async def one(q: dict, k: int) -> None:
        async with (sem_answers if k == 0 else sem_resamples):
            if state["stopped"] or q["id"] in state["refused"]:
                return
            if k > 0:  # resamples only for questions whose answer (sample 0) came back fine
                first = cached(cache_dir / q["id"] / "0.json")
                if not (first and first.get("ok")):
                    return
            reserve = (state["spent"] / max(1, done + state["new"])) * cfg["api"]["max_concurrency"]
            if state["spent"] + reserve > budget:
                state["stopped"] = state["budget_hit"] = True
                return
            body = {"model": model, "messages": messages_for(cfg, q["question"]),
                    "temperature": g["temperature"], "reasoning_effort": g["reasoning_effort"],
                    "max_completion_tokens": g["max_completion_tokens"]}
            t = time.monotonic()
            try:
                r = await call_with_retry(lambda: client.chat.completions.create(**body), limiter,
                                          cfg["api"]["max_retries"])
            except Exception as exc:  # noqa: BLE001
                info = error_info(exc)
                state["errors"] += 1
                err_log.write(json.dumps({"id": q["id"], "k": k, **info}) + "\n")
                err_log.flush()
                msg = (info.get("body") or info.get("message") or "")[:300]
                if is_content_flag(info):
                    state["refused"].add(q["id"])
                    state["flags"] += 1
                    ref_log.write(json.dumps({"id": q["id"], "question": q["question"],
                                              "at": datetime.now(timezone.utc).isoformat()}) + "\n")
                    ref_log.flush()
                    print(f"  content filter rejected {q['id']}: {q['question'][:100]}", flush=True)
                    if state["flags"] >= max_flags:
                        stop(f"{state['flags']} question(s) rejected by the content filter; add the topic to "
                             "screen.py before continuing")
                    return
                key = (info.get("status"), msg[:80])
                if key not in state["seen"] and len(state["seen"]) < 5:
                    state["seen"].add(key)
                    print(f"  error: HTTP {info.get('status')} {msg}", flush=True)
                state["streak"] += 1
                if info.get("status") in (401, 402, 403) or any(w in msg.lower() for w in ACCOUNT_WORDS):
                    stop(f"HTTP {info.get('status')}: {msg}")
                elif state["streak"] >= 20:
                    stop(f"20 failed requests in a row; last was HTTP {info.get('status')}: {msg}")
                return
            state["streak"] = 0
            ch = r.choices[0]
            u = usage_dict(r.usage)
            c = cost_usd(cfg, model, u)
            prev = cached(cache_dir / q["id"] / f"{k}.json")   # a failed earlier attempt being retried
            prior_cost = prev.get("cost_usd", 0) if prev else 0
            content = (ch.message.content or "").strip()
            good = bool(content) and ch.finish_reason == "stop"
            write_json_atomic(cache_dir / q["id"] / f"{k}.json", {
                "id": q["id"], "k": k, "split": q["split"], "group": q["group"], "ok": good,
                "content": content, "finish_reason": ch.finish_reason, "usage": u,
                "cost_usd": c + prior_cost,                      # includes earlier failed attempts
                "attempts": (prev.get("attempts", 1) + 1) if prev else 1,
                "latency_s": round(time.monotonic() - t, 2), "model": r.model or model,
                "settings_hash": h, "prompt_version": cfg["prompts"]["version"],
                "created_at": datetime.now(timezone.utc).isoformat()})
            state["spent"] += c
            state["new"] += 1
            state["ok" if good else "bad"] += 1
            if state["new"] % 50 == 0:
                rate = state["new"] / (time.monotonic() - t_start)
                eta = (len(todo) - state["new"]) / rate / 60 if rate else 0
                print(f"{state['new']}/{len(todo)} · ok {state['ok']} · cut off/empty {state['bad']} · "
                      f"flags {state['flags']} · errors {state['errors']} · spent {fmt_money(cfg, state['spent'])} · "
                      f"ETA {eta:.0f} min", flush=True)

    # phase 1: answers (sample 0) for every question; phase 2: resamples
    await asyncio.gather(*[one(q, k) for q, k in todo if k == 0])
    await asyncio.gather(*[one(q, k) for q, k in todo if k > 0])
    err_log.close()
    ref_log.close()
    print(f"\nfinished {state['new']} new samples in {(time.monotonic() - t_start) / 60:.1f} min · "
          f"ok {state['ok']} · cut off/empty {state['bad']} (rerun with --retry-failed) · "
          f"content flags {state['flags']} · errors {state['errors']} (see logs/generate_errors.jsonl) · "
          f"total spent {fmt_money(cfg, state['spent'])}")
    if state["stop_reason"]:
        print(f"STOPPED EARLY: {state['stop_reason']}. Fix the cause, then run the same command to resume; "
              "finished samples are kept.")
    elif state["budget_hit"]:
        print("STOPPED at the budget cap. Raise budgets_usd.generate in config.yaml to continue; "
              "finished samples are kept.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped. Finished samples are saved; run the same command again to resume.")
