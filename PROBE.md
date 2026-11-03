# API probe — 2026-10-26 20:27 UTC

Model: `muse-spark-1.3` · settings hash `70497e10404d` · base URL `https://api.meta.ai/v1`

## Access

- Models listed: muse-image-1.0, muse-spark-1.1, muse-spark-1.2, muse-spark-1.2-contributor, muse-spark-1.3, muse-spark-1.3-contributor, muse-voice-transcribe-1.0, muse-wonder-1.0, sam-3.1
- Basic call on configured model: accepted, 6.02 s
- Basic call on standard model: not run
- Credit balance before probe: ______  after: ______  (read from the dashboard)

## Parameters

| Probe | Result | Reasoning tokens | Latency |
|---|---|---|---|
| `effort_none` | rejected (HTTP 400): {"error":{"code":null,"message":"reasoning_effort 'none' is not supported for model 'muse-spark-1.3'. Supported values: [minimal, low, medium, high, xhigh, max] | – | 0.94 s |
| `effort_minimal` | accepted | 462 | 4.89 s |
| `effort_low` | accepted | 1195 | 6.77 s |
| `logprobs_none` | not run | – | – |
| `logprobs_minimal` | rejected (HTTP 400): {"error":{"code":null,"message":"`logprobs` cannot be true when `reasoning_effort` is `minimal`. logprobs is not supported with reasoning.","param":"logprobs"," | – | 1.12 s |
| `n_2` | rejected (HTTP 400): {"error":{"code":null,"message":"`n` must be 1, got 2","param":"n","type":"invalid_request_error"}} | – | 0.27 s |

## Resample diversity

- 5 samples at temperature 1.0: 5 unique of 5, mean pairwise word overlap 0.42 (min 0.355)
- Two calls at temperature 0 with a seed identical: False

## Structured output

- JSON-schema claim extraction: accepted; parsed: True, 27 claims
- Example: `[{"confidence": 90, "text": "Taking ibuprofen regularly while taking lisinopril can raise blood pressure."}, {"confidence": 88, "text": "Taking ibuprofen regularly while taking lisinopril can make lisinopril less effective."}, {"confidence": 92, "text": "Nonsteroidal anti-inflammatory drugs cause so`

## web_search (Responses API)

- default context: status completed, 8.14 s, 3 tool actions {'open_page': 1, 'search': 2}, 4 citations, 9717 input tokens
  - cost of this call: $0.0215 (R0.36) if only searches are billed, $0.0240 (R0.40) if page opens are billed too (raw response: logs/probe_web_search.json)
- `search_context_size: low`: status completed, 8.84 s, 3 tool actions {'search': 3}, 0 citations, 7026 input tokens
  - cost of this call: $0.0212 (R0.35) if only searches are billed, $0.0212 (R0.35) if page opens are billed too (raw response: logs/probe_web_search_low.json)
- If no `search` action appears, the model chose not to search: never assume it did.

## Concurrency

- 10 parallel calls: 10 ok in 3.08 s; errors []

## Cost projection (1,300 questions: 6 samples + 1 extraction each)

- Mean reasoning tokens per sample: 288; mean latency 4.5 s

| Model | Per question | 1,300 questions | With reruns (×1.6) |
|---|---|---|---|
| `muse-spark-1.3` | $0.0175 (R0.29) | $22.8051 (R379.48) | $36.4882 (R607.16) |
| `muse-spark-1.3-contributor` | $0.0009 (R0.01) | $1.1185 (R18.61) | $1.7895 (R29.78) |

## Decisions

- [ ] `generation.reasoning_effort`: lowest accepted value is **minimal**
- [ ] Logprobs: not available → black-box scores + Glimmer
- [ ] n>1: rejected → one request per sample (already the plan)
- [ ] Diversity: OK
- [ ] Budget: the 'with reruns' figure for your model must fit `budgets_usd.generate` ($45); if not, lower `samples_per_question` to 5 or `max_completion_tokens`
- [ ] Freeze config.yaml and commit it with this file
