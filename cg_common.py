"""Shared helpers: config, Meta Model API client, rate limiting, retries, cost accounting."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path = "config.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def settings_hash(cfg: dict) -> str:
    """Fingerprint of everything that changes what the model is asked; stored with every answer."""
    frozen = {"model": cfg["api"]["model"], "generation": cfg["generation"], "prompts": cfg["prompts"]}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:12]


def messages_for(cfg: dict, question: str) -> list[dict]:
    pr = cfg["prompts"]
    return [{"role": pr.get("instruction_role", "developer"), "content": pr["system"].strip()},
            {"role": "user", "content": pr["user"].format(question=question)}]


def make_client(cfg: dict):
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    load_dotenv(ROOT / ".env")
    key = os.getenv(cfg["api"]["api_key_env"]) or os.getenv("META_API_KEY")
    if not key:
        raise SystemExit(f"{cfg['api']['api_key_env']} is not set. Copy .env.example to .env and add your key.")
    base = os.getenv("META_BASE_URL", cfg["api"]["base_url"])
    return AsyncOpenAI(api_key=key, base_url=base, timeout=cfg["api"]["timeout_s"], max_retries=0)


# ----------------------------------------------------------------------------- usage and cost
def usage_dict(usage: Any) -> dict:
    """Normalise a chat-completions or responses `usage` object into plain token counts."""
    if usage is None:
        return {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0}
    u = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    inp = u.get("prompt_tokens", u.get("input_tokens")) or 0
    out = u.get("completion_tokens", u.get("output_tokens")) or 0
    pdet = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    cdet = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
    return {"input": inp, "cached_input": (pdet or {}).get("cached_tokens") or 0,
            "output": out, "reasoning": (cdet or {}).get("reasoning_tokens") or 0}


def cost_usd(cfg: dict, model: str, u: dict, searches: int = 0) -> float:
    price = cfg["pricing_usd_per_million"].get(model)
    if price is None:
        raise KeyError(f"No price for model {model!r} in config.yaml")
    uncached = max(u["input"] - u["cached_input"], 0)
    return (uncached * price["input"] + u["cached_input"] * price["cached_input"]
            + u["output"] * price["output"]) / 1e6 + searches * cfg["web_search_usd_per_query"]


def fmt_money(cfg: dict, usd: float) -> str:
    return f"${usd:,.4f} (R{usd * cfg['usd_to_zar']:,.2f})"


# ----------------------------------------------------------------------------- rate limit and retry
class RateLimiter:
    """Spaces request starts evenly so we stay under `rpm` requests per minute."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


def _retryable(exc: Exception) -> tuple[bool, float | None]:
    import openai

    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return True, None
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        retry_after = None
        try:
            retry_after = float(exc.response.headers.get("retry-after"))
        except (TypeError, ValueError, AttributeError):
            pass
        return status == 429 or status >= 500, retry_after
    return False, None


async def call_with_retry(fn: Callable[[], Awaitable[Any]], limiter: RateLimiter, max_retries: int) -> Any:
    for attempt in range(max_retries + 1):
        await limiter.wait()
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            retry, after = _retryable(exc)
            if not retry or attempt == max_retries:
                raise
            await asyncio.sleep(after if after is not None else min(60, 2 ** attempt) + random.random())


def error_info(exc: Exception) -> dict:
    status = getattr(exc, "status_code", None)
    body = None
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = resp.text[:500]
        except Exception:  # noqa: BLE001
            body = None
    return {"type": type(exc).__name__, "status": status, "message": str(exc)[:500], "body": body}


def write_json_atomic(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)
