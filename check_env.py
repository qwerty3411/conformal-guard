"""Check that local environment for the conformal hallucination guard works.

    python check_env.py            # imports, device, LlamaFirewall hook, NLTK data
    python check_env.py --models   # also download and run the 3 local scoring models 
    python check_env.py --api      # also make one call to the Meta Model API (needs META_API_KEY)
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
import time

OK, FAIL, WARN = "\033[32mOK\033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"
failures: list[str] = []


def report(status: str, name: str, detail: str = "") -> None:
    print(f"[{status}] {name}" + (f"  {detail}" if detail else ""))
    if status == FAIL:
        failures.append(name)


def check_python() -> None:
    v = sys.version_info
    if (3, 11) <= (v.major, v.minor) <= (3, 13):
        report(OK, "Python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        report(FAIL, "Python", f"{v.major}.{v.minor} found; use 3.12 (3.11-3.13 work)")
    if sys.prefix == sys.base_prefix:
        report(WARN, "venv", "not running inside a virtual environment")
    else:
        report(OK, "venv", sys.prefix)


def check_imports() -> None:
    modules = [
        ("openai", "openai"), ("modal", "modal"), ("httpx", "httpx"),
        ("tenacity", "tenacity"), ("dotenv", "python-dotenv"), ("yaml", "pyyaml"),
        ("torch", "torch"), ("transformers", "transformers"), ("accelerate", "accelerate"),
        ("sentence_transformers", "sentence-transformers"), ("datasets", "datasets"),
        ("nltk", "nltk"), ("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"),
        ("scipy", "scipy"), ("sklearn", "scikit-learn"), ("mapie", "mapie"),
        ("matplotlib", "matplotlib"), ("plotly", "plotly"), ("llamafirewall", "llamafirewall"),
        ("synthetic_data_kit", "synthetic-data-kit"), ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"), ("pydantic", "pydantic"), ("gradio", "gradio"),
        ("pytest", "pytest"),
    ]
    from importlib.metadata import PackageNotFoundError, version

    for mod, pkg in modules:
        try:
            importlib.import_module(mod)
            try:
                ver = version(pkg)
            except PackageNotFoundError:
                ver = ""
            report(OK, pkg, ver)
        except Exception as e:  # noqa: BLE001
            report(FAIL, pkg, f"{type(e).__name__}: {e}")
    try:
        from mapie.risk_control import BinaryClassificationController  # noqa: F401
        report(OK, "mapie.risk_control", "Learn-Then-Test / risk control available")
    except Exception as e:  # noqa: BLE001
        report(WARN, "mapie.risk_control", f"{type(e).__name__}: {e} (check MAPIE docs for the module name)")


def check_nltk() -> None:
    import nltk
    for res in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{res}")
            report(OK, f"nltk {res}")
        except LookupError:
            ok = nltk.download(res, quiet=True)
            report(OK if ok else FAIL, f"nltk {res}", "downloaded" if ok else
                   "download failed: check your internet connection, or run "
                   "`python -m nltk.downloader punkt punkt_tab` (behind a proxy, see NLTK's message above)")
    try:
        import minicheck.minicheck  # noqa: F401  (import also triggers its own punkt check)
        report(OK, "minicheck", "import works (use model_name='flan-t5-large')")
    except Exception as e:  # noqa: BLE001
        report(FAIL, "minicheck", f"{type(e).__name__}: {e}")


def device() -> str:
    import torch
    if torch.cuda.is_available():
        dev, name = "cuda", torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev, name = "mps", "Apple GPU"
    else:
        dev, name = "cpu", "no GPU found: local scoring will be slower but works"
    report(OK, "torch device", f"{dev} ({name})")
    return dev


def check_llamafirewall() -> None:
    """Register a dummy custom scanner and run it on an assistant message.

    Confirms the hook the real ConformalFactualityScanner will use. Notes:
    - custom scanners are created with NO constructor arguments, so load the
      conformal threshold from a file or env var inside __init__;
    - inside FastAPI (a running event loop) call `await fw.scan_async(...)`,
      not `fw.scan(...)`, which uses asyncio.run().
    """
    from llamafirewall import (AssistantMessage, LlamaFirewall, Role, ScanDecision,
                               Scanner, ScanResult, register_llamafirewall_scanner)

    @register_llamafirewall_scanner("env_check_scanner")
    class EnvCheckScanner(Scanner):
        def __init__(self) -> None:
            super().__init__(scanner_name="env_check_scanner")

        async def scan(self, message, past_trace=None) -> ScanResult:
            flagged = "REVIEW" in message.content
            return ScanResult(
                decision=ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED if flagged else ScanDecision.ALLOW,
                reason="dummy check", score=0.9 if flagged else 0.1)

    fw = LlamaFirewall(scanners={Role.ASSISTANT: ["env_check_scanner"]})
    r1 = fw.scan(AssistantMessage(content="Ibuprofen is an NSAID."))
    r2 = asyncio.run(fw.scan_async(AssistantMessage(content="REVIEW this claim")))
    if r1.decision == ScanDecision.ALLOW and r2.decision == ScanDecision.HUMAN_IN_THE_LOOP_REQUIRED:
        report(OK, "LlamaFirewall custom scanner", "register + scan + scan_async work")
    else:
        report(FAIL, "LlamaFirewall custom scanner", f"unexpected decisions {r1.decision}, {r2.decision}")


def check_models(dev: str) -> None:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    premise = "Ibuprofen is a nonsteroidal anti-inflammatory drug used to reduce fever and pain."
    claim_true, claim_false = "Ibuprofen is an NSAID.", "Ibuprofen is an antibiotic."

    t = time.time()
    name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        nli = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval()
        labels = {v.lower(): k for k, v in nli.config.id2label.items()}
        with torch.no_grad():
            enc = tok([premise, premise], [claim_true, claim_false], return_tensors="pt",
                      padding=True, truncation=True).to(dev)
            probs = nli(**enc).logits.softmax(-1)[:, labels["entailment"]].tolist()
        good = probs[0] > 0.5 > probs[1]
        report(OK if good else WARN, "DeBERTa NLI",
               f"entail(true)={probs[0]:.2f} entail(false)={probs[1]:.2f}  {time.time()-t:.0f}s")
    except Exception as e:  # noqa: BLE001
        report(FAIL, "DeBERTa NLI", f"{type(e).__name__}: {e}")

    t = time.time()
    try:
        from minicheck.minicheck import MiniCheck
        mc = MiniCheck(model_name="flan-t5-large", cache_dir=None)
        _, probs, _, _ = mc.score(docs=[premise, premise], claims=[claim_true, claim_false])
        good = probs[0] > 0.5 > probs[1]
        report(OK if good else WARN, "MiniCheck-Flan-T5-Large",
               f"p(true)={probs[0]:.2f} p(false)={probs[1]:.2f}  {time.time()-t:.0f}s")
    except Exception as e:  # noqa: BLE001
        report(FAIL, "MiniCheck-Flan-T5-Large", f"{type(e).__name__}: {e}")

    t = time.time()
    try:
        emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)
        v = emb.encode(["What is the max daily dose of ibuprofen?", "How do I file my taxes?"],
                       normalize_embeddings=True)
        report(OK, "bge-small embeddings", f"dim={v.shape[1]}  {time.time()-t:.0f}s")
    except Exception as e:  # noqa: BLE001
        report(FAIL, "bge-small embeddings", f"{type(e).__name__}: {e}")


def check_api() -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    key = os.getenv("META_API_KEY")
    if not key:
        report(FAIL, "Meta Model API", "META_API_KEY not set (copy .env.example to .env)")
        return
    base = os.getenv("META_BASE_URL", "https://api.meta.ai/v1")
    model = os.getenv("META_MODEL", "muse-spark-1.3")
    client = OpenAI(api_key=key, base_url=base)
    t = time.time()
    try:
        r = client.chat.completions.create(
            model=model, reasoning_effort="minimal",
            messages=[{"role": "user", "content": "Reply with exactly: ok"}])
        text = (r.choices[0].message.content or "").strip()
        report(OK, "Meta Model API", f"{model} replied {text!r} in {time.time()-t:.1f}s; usage={r.usage}")
    except Exception as e:  # noqa: BLE001
        report(FAIL, "Meta Model API", f"{type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true", help="download and run local scoring models")
    ap.add_argument("--api", action="store_true", help="make one Meta Model API call")
    args = ap.parse_args()

    check_python()
    check_imports()
    check_nltk()
    dev = device()
    try:
        check_llamafirewall()
    except Exception as e:  # noqa: BLE001
        report(FAIL, "LlamaFirewall custom scanner", f"{type(e).__name__}: {e}")
    if args.models:
        check_models(dev)
    if args.api:
        check_api()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
