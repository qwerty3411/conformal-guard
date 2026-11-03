from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import time
from datetime import datetime, timezone

from cg_common import (RateLimiter, call_with_retry, cost_usd, error_info, fmt_money, load_config,
                       make_client, messages_for, path, settings_hash, usage_dict)

QUESTION = "Can I take ibuprofen while I am on lisinopril for high blood pressure?"
CLAIM_SCHEMA = {
    "name": "claims",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims"],
        "properties": {"claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["text", "confidence"],
            "properties": {"text": {"type": "string"},
                           "confidence": {"type": "integer", "minimum": 0, "maximum": 100}}}}},
    },
}
EXTRACT_PROMPT = ("Split the answer below into atomic, self-contained factual claims (one fact each, "
                  "pronouns resolved). For each claim give your confidence from 0 to 100 that it is true.\n\n"
                  "Answer:\n{answer}")


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / max(1, len(ta | tb))


class Probe:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.client = make_client(cfg)
        self.limiter = RateLimiter(cfg["api"]["rpm_limit"])
        self.model = cfg["api"]["model"]
        self.results: dict = {}

    async def chat(self, label: str, model: str | None = None, retries: int = 1, **kw) -> dict:
        model = model or self.model
        body = {"model": model, "messages": kw.pop("messages", messages_for(self.cfg, QUESTION)), **kw}
        t = time.monotonic()
        try:
            r = await call_with_retry(lambda: self.client.chat.completions.create(**body), self.limiter, retries)
            ch = r.choices[0]
            u = usage_dict(r.usage)
            lp = getattr(ch, "logprobs", None)
            rec = {"ok": True, "latency_s": round(time.monotonic() - t, 2), "model": model, "usage": u,
                   "cost_usd": cost_usd(self.cfg, model, u) if model in self.cfg["pricing_usd_per_million"] else None,
                   "finish_reason": ch.finish_reason, "content": ch.message.content,
                   "reasoning_chars": len(getattr(ch.message, "reasoning_content", None) or ""),
                   "logprobs_returned": bool(lp and getattr(lp, "content", None))}
        except Exception as exc:  # noqa: BLE001
            rec = {"ok": False, "latency_s": round(time.monotonic() - t, 2), "model": model, "error": error_info(exc)}
        self.results[label] = {"request": {k: v for k, v in body.items() if k != "messages"}, **rec}
        return rec

    async def run(self, skip_standard: bool) -> None:
        g = self.cfg["generation"]
        base = {"temperature": g["temperature"], "max_completion_tokens": g["max_completion_tokens"]}

        # 1. access: list models, one basic call per tier
        try:
            models = await self.client.models.list()
            self.results["models"] = {"ok": True, "ids": sorted(m.id for m in models.data)}
        except Exception as exc:  # noqa: BLE001
            self.results["models"] = {"ok": False, "error": error_info(exc)}
        await self.chat("basic_configured", reasoning_effort=g["reasoning_effort"], **base)
        if not skip_standard and self.model != "muse-spark-1.3":
            await self.chat("basic_standard", model="muse-spark-1.3", reasoning_effort="minimal",
                            max_completion_tokens=400)

        # 2. reasoning effort floor
        for eff in ("none", "minimal", "low"):
            await self.chat(f"effort_{eff}", reasoning_effort=eff, **base)

        # 3. logprobs, with the lowest efforts
        for eff in ("none", "minimal"):
            if self.results.get(f"effort_{eff}", {}).get("ok"):
                await self.chat(f"logprobs_{eff}", reasoning_effort=eff, logprobs=True, top_logprobs=5, **base)

        # 4. n > 1
        await self.chat("n_2", reasoning_effort=g["reasoning_effort"], n=2, **base)

        # 5. diversity: 5 resamples at the configured temperature; determinism at T=0
        samples = await asyncio.gather(*[
            self.chat(f"sample_{i}", reasoning_effort=g["reasoning_effort"], **base) for i in range(5)])
        texts = [s.get("content") or "" for s in samples if s.get("ok")]
        pairs = [jaccard(a, b) for i, a in enumerate(texts) for b in texts[i + 1:]]
        self.results["diversity"] = {
            "n_ok": len(texts), "unique_texts": len(set(texts)),
            "mean_pairwise_jaccard": round(statistics.mean(pairs), 3) if pairs else None,
            "min_pairwise_jaccard": round(min(pairs), 3) if pairs else None}
        t0 = await asyncio.gather(*[self.chat(f"t0_{i}", reasoning_effort=g["reasoning_effort"],
                                              temperature=0, seed=7,
                                              max_completion_tokens=g["max_completion_tokens"]) for i in range(2)])
        self.results["determinism_t0"] = {"identical": (t0[0].get("content") == t0[1].get("content"))
                                          if all(x.get("ok") for x in t0) else None}

        # 6. structured output: claim extraction with confidence
        answer = texts[0] if texts else "Ibuprofen can raise blood pressure."
        ext = await self.chat("extract_json_schema", reasoning_effort=g["reasoning_effort"],
                              messages=[{"role": "user", "content": EXTRACT_PROMPT.format(answer=answer)}],
                              response_format={"type": "json_schema", "json_schema": CLAIM_SCHEMA},
                              max_completion_tokens=g["max_completion_tokens"])
        parsed = None
        if ext.get("ok"):
            try:
                parsed = json.loads(ext["content"])["claims"]
            except Exception:  # noqa: BLE001
                parsed = None
        self.results["extract_parsed"] = {"ok": parsed is not None, "n_claims": len(parsed) if parsed else 0,
                                          "example": parsed[:3] if parsed else None}

        # 7. web_search through the Responses API (default context, then the cheaper "low" context)
        for label, tool in (("web_search", {"type": "web_search"}),
                            ("web_search_low", {"type": "web_search", "search_context_size": "low"})):
            await self.web_search(label, tool, g)

        await self.tail(g, samples, ext)

    async def web_search(self, label: str, tool: dict, g: dict) -> None:
        t = time.monotonic()
        try:
            r = await call_with_retry(lambda: self.client.responses.create(
                model=self.model, input=QUESTION, tools=[tool],
                reasoning={"effort": g["reasoning_effort"]}), self.limiter, 1)
            raw = r.model_dump()
            path(self.cfg, "logs").mkdir(parents=True, exist_ok=True)
            (path(self.cfg, "logs") / f"probe_{label}.json").write_text(json.dumps(raw, indent=1))
            calls = [o for o in raw.get("output", []) if o.get("type") == "web_search_call"]
            actions = [((o.get("action") or {}).get("type") or "unknown") for o in calls]
            cites = [a for o in raw.get("output", []) if o.get("type") == "message"
                     for c in o.get("content", []) for a in (c.get("annotations") or [])
                     if a.get("type") == "url_citation"]
            u = usage_dict(raw.get("usage"))
            n_search = actions.count("search")
            self.results[label] = {
                "ok": True, "latency_s": round(time.monotonic() - t, 2), "status": raw.get("status"),
                "web_search_calls": len(calls), "actions": {a: actions.count(a) for a in set(actions)},
                "url_citations": len(cites), "example_citations": [{"url": c.get("url"), "title": c.get("title")}
                                                                    for c in cites[:3]],
                "usage": u, "cost_usd_if_searches_billed": cost_usd(self.cfg, self.model, u, n_search),
                "cost_usd_if_all_actions_billed": cost_usd(self.cfg, self.model, u, len(calls))}
        except Exception as exc:  # noqa: BLE001
            self.results[label] = {"ok": False, "error": error_info(exc)}

    async def tail(self, g: dict, samples: list, ext: dict) -> None:
        # 8. concurrency: 10 parallel short calls
        t = time.monotonic()
        conc = await asyncio.gather(*[self.chat(f"conc_{i}", reasoning_effort=g["reasoning_effort"],
                                                max_completion_tokens=400,
                                                messages=[{"role": "user", "content": "Reply with: ok"}])
                                      for i in range(10)])
        self.results["concurrency"] = {"wall_s": round(time.monotonic() - t, 2),
                                       "ok": sum(1 for c in conc if c.get("ok")),
                                       "errors": [c["error"]["status"] for c in conc if not c.get("ok")]}

        # 9. cost projection from the real samples + extraction
        ok_samples = [s for s in samples if s.get("ok")]
        if ok_samples and ext.get("ok"):
            gen_in = statistics.mean(s["usage"]["input"] for s in ok_samples)
            gen_out = statistics.mean(s["usage"]["output"] for s in ok_samples)
            per_q = {"input": g["samples_per_question"] * gen_in + ext["usage"]["input"],
                     "cached_input": 0,
                     "output": g["samples_per_question"] * gen_out + ext["usage"]["output"]}
            proj = {}
            for m in self.cfg["pricing_usd_per_million"]:
                c = cost_usd(self.cfg, m, per_q)
                proj[m] = {"per_question_usd": c, "x1300_usd": c * 1300, "x1300_with_reruns_usd": c * 1300 * 1.6}
            self.results["projection"] = {"tokens_per_question": per_q, "by_model": proj,
                                          "mean_reasoning_tokens_per_sample":
                                              statistics.mean(s["usage"]["reasoning"] for s in ok_samples),
                                          "mean_latency_s": statistics.mean(s["latency_s"] for s in ok_samples)}


def ok(r: dict | None) -> str:
    if not r:
        return "not run"
    if r.get("ok"):
        return "accepted"
    e = r.get("error", {})
    return f"rejected (HTTP {e.get('status')}): {(e.get('body') or e.get('message') or '')[:160]}"


def write_report(cfg: dict, res: dict) -> str:
    g = cfg["generation"]
    L = [f"# API probe — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", "",
         f"Model: `{cfg['api']['model']}` · settings hash `{settings_hash(cfg)}` · "
         f"base URL `{os.getenv('META_BASE_URL', cfg['api']['base_url'])}`", "",
         "## Access", "",
         f"- Models listed: {', '.join(res['models'].get('ids', [])) or ok(res['models'])}",
         f"- Basic call on configured model: {ok(res.get('basic_configured'))}"
         + (f", {res['basic_configured']['latency_s']} s" if res.get("basic_configured", {}).get("ok") else ""),
         f"- Basic call on standard model: {ok(res.get('basic_standard'))}",
         "- Credit balance before probe: ______  after: ______  (read from the dashboard)", "",
         "## Parameters", "", "| Probe | Result | Reasoning tokens | Latency |", "|---|---|---|---|"]
    for k in ("effort_none", "effort_minimal", "effort_low", "logprobs_none", "logprobs_minimal", "n_2"):
        r = res.get(k)
        extra = ""
        if r and r.get("ok") and k.startswith("logprobs"):
            extra = " · logprobs returned" if r.get("logprobs_returned") else " · accepted but no logprobs in response"
        L.append(f"| `{k}` | {ok(r)}{extra} | {r['usage']['reasoning'] if r and r.get('ok') else '–'} | "
                 f"{str(r['latency_s']) + ' s' if r else '–'} |")
    d, ws, cc = res.get("diversity", {}), res.get("web_search", {}), res.get("concurrency", {})
    L += ["", "## Resample diversity", "",
          f"- 5 samples at temperature {g['temperature']}: {d.get('unique_texts')} unique of {d.get('n_ok')}, "
          f"mean pairwise word overlap {d.get('mean_pairwise_jaccard')} (min {d.get('min_pairwise_jaccard')})",
          f"- Two calls at temperature 0 with a seed identical: {res.get('determinism_t0', {}).get('identical')}", "",
          "## Structured output", "",
          f"- JSON-schema claim extraction: {ok(res.get('extract_json_schema'))}; parsed: "
          f"{res.get('extract_parsed', {}).get('ok')}, {res.get('extract_parsed', {}).get('n_claims')} claims",
          f"- Example: `{json.dumps(res.get('extract_parsed', {}).get('example'), ensure_ascii=False)[:300]}`", "",
          "## web_search (Responses API)", ""]
    for label in ("web_search", "web_search_low"):
        w = res.get(label, {})
        name = "default context" if label == "web_search" else "`search_context_size: low`"
        if w.get("ok"):
            L += [f"- {name}: status {w['status']}, {w['latency_s']} s, {w['web_search_calls']} tool actions "
                  f"{w['actions']}, {w['url_citations']} citations, {w['usage']['input']} input tokens",
                  f"  - cost of this call: {fmt_money(cfg, w['cost_usd_if_searches_billed'])} if only searches are "
                  f"billed, {fmt_money(cfg, w['cost_usd_if_all_actions_billed'])} if page opens are billed too "
                  f"(raw response: logs/probe_{label}.json)"]
        else:
            L.append(f"- {name}: {ok(w)}")
    L.append("- If no `search` action appears, the model chose not to search: never assume it did.")
    L += ["", "## Concurrency", "", f"- 10 parallel calls: {cc.get('ok')} ok in {cc.get('wall_s')} s; errors {cc.get('errors')}"]
    p = res.get("projection")
    if p:
        L += ["", "## Cost projection (1,300 questions: 6 samples + 1 extraction each)", "",
              f"- Mean reasoning tokens per sample: {p['mean_reasoning_tokens_per_sample']:.0f}; "
              f"mean latency {p['mean_latency_s']:.1f} s", "",
              "| Model | Per question | 1,300 questions | With reruns (×1.6) |", "|---|---|---|---|"]
        for m, v in p["by_model"].items():
            L.append(f"| `{m}` | {fmt_money(cfg, v['per_question_usd'])} | {fmt_money(cfg, v['x1300_usd'])} | "
                     f"{fmt_money(cfg, v['x1300_with_reruns_usd'])} |")
    # decisions
    floor = next((e for e in ("none", "minimal", "low") if res.get(f"effort_{e}", {}).get("ok")), None)
    lp = [e for e in ("none", "minimal") if res.get(f"logprobs_{e}", {}).get("logprobs_returned")]
    L += ["", "## Decisions", "",
          f"- [ ] `generation.reasoning_effort`: lowest accepted value is **{floor}**",
          f"- [ ] Logprobs: {'available at ' + ', '.join(lp) + ' (bonus score feature)' if lp else 'not available → black-box scores + Glimmer'}",
          f"- [ ] n>1: {'works' if res.get('n_2', {}).get('ok') else 'rejected → one request per sample (already the plan)'}",
          f"- [ ] Diversity: {'OK' if (d.get('mean_pairwise_jaccard') or 1) < 0.85 and d.get('unique_texts', 0) >= 4 else 'LOW → raise temperature before the full run'}",
          "- [ ] Budget: the 'with reruns' figure for your model must fit `budgets_usd.generate` "
          f"(${cfg['budgets_usd']['generate']}); if not, lower `samples_per_question` to 5 or `max_completion_tokens`",
          "- [ ] Freeze config.yaml and commit it with this file"]
    return "\n".join(L) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--skip-standard", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pr = Probe(cfg)
    await pr.run(args.skip_standard)
    logs = path(cfg, "logs")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "probe_results.json").write_text(json.dumps(pr.results, indent=1, ensure_ascii=False, default=str))
    report = write_report(cfg, pr.results)
    (path(cfg, "logs").parent / "PROBE.md").write_text(report, encoding="utf-8")
    spent = sum(v.get("cost_usd") or 0 for v in pr.results.values() if isinstance(v, dict))
    spent += sum(pr.results.get(k, {}).get("cost_usd_if_all_actions_billed", 0) or 0
                 for k in ("web_search", "web_search_low"))
    print(report)
    print(f"Probe spend (approx.): {fmt_money(cfg, spent)}")


if __name__ == "__main__":
    asyncio.run(main())
