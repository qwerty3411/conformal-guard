"""Builds local evidence index: MedlinePlus health-topic summaries, embedded and indexed with FAISS.

    python build_index.py              # download (once), chunk, embed, index  (~5-15 min on CPU)
    python build_index.py --search "Can ibuprofen raise blood pressure?"   

Evidence score looks up each claim instead of calling a web-search API, so the evidence is
free, and is identical in calibration and in the demo. The source file is pinned
by date (config.yaml -> evidence.medlineplus_file); never rebuild with a different file after
calibration.

Writes data/evidence/: passages.jsonl, embeddings.npy, index.faiss (if FAISS loads), meta.json.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib

import numpy as np

from cg_common import ROOT, load_config
from label_claims import split_sentences

BASE = "https://medlineplus.gov/xml/"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge retrieval instruction


def clean_html(raw: str) -> str:
    text = re.sub(r"<\s*(br|/p|/li|/ul|/h\d)\s*/?>", ". ", raw, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s*\.\s*(\.\s*)+", ". ", text)
    return re.sub(r"\s+", " ", text).strip()


def download(fname: str, dest_dir) -> bytes:
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / fname
    if not local.exists():
        print(f"downloading {BASE}{fname}")
        req = urllib.request.Request(BASE + fname, headers={"User-Agent": "conformal-guard/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            local.write_bytes(r.read())
    return local.read_bytes()


def parse_topics(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    topics = []
    for t in root.iter("health-topic"):
        if (t.get("language") or "English") != "English":
            continue
        fs = t.find("full-summary")
        if fs is None:
            continue
        raw = (fs.text or "") + "".join(ET.tostring(child, encoding="unicode") for child in fs)
        summary = clean_html(raw)
        if summary:
            topics.append({"title": t.get("title") or "", "url": t.get("url") or "", "summary": summary,
                           "also_called": [a.text for a in t.findall("also-called") if a.text]})
    return topics


def chunk(topics: list[dict], size: int, overlap: int) -> list[dict]:
    passages = []
    for t in topics:
        sents = [s for s, _, _ in split_sentences(t["summary"], min_words=3)]
        step = max(1, size - overlap)
        for i in range(0, max(1, len(sents) - overlap), step):
            body = " ".join(sents[i:i + size])
            if body:
                passages.append({"title": t["title"], "url": t["url"], "text": f"{t['title']}: {body}"})
    return passages


class Embedder:
    def __init__(self, model_name: str, fake: bool = False):
        self.fake = fake
        if not fake:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str], query: bool = False) -> np.ndarray:
        if self.fake:  # hashed bag-of-words stand-in for tests
            v = np.zeros((len(texts), 384), dtype=np.float32)
            for i, t in enumerate(texts):
                for w in re.findall(r"[a-z]+", t.lower()):
                    v[i, zlib.crc32(w.encode()) % 384] += 1
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=not query, convert_to_numpy=True).astype(np.float32)


class EvidenceIndex:
    """Load the built index and return the top-k passages for queries (FAISS if available, else numpy)."""

    def __init__(self, cfg: dict, fake: bool = False):
        d = ROOT / cfg["evidence"]["dir"]
        self.passages = [json.loads(line) for line in (d / "passages.jsonl").read_text(encoding="utf-8").splitlines()]
        self.emb = np.load(d / "embeddings.npy")
        self.embedder = Embedder(cfg["evidence"]["embedding_model"], fake=fake)
        self.faiss = None
        try:
            import faiss
            if (d / "index.faiss").exists():
                self.faiss = faiss.read_index(str(d / "index.faiss"))
        except Exception:  # noqa: BLE001  (missing or blocked by Smart App Control: numpy fallback)
            self.faiss = None

    def search(self, queries: list[str], k: int) -> list[list[tuple[float, dict]]]:
        q = self.embedder(queries, query=True)
        if self.faiss is not None:
            scores, idx = self.faiss.search(q, k)
        else:
            sims = q @ self.emb.T
            idx = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, idx, axis=1)
        return [[(float(s), self.passages[i]) for s, i in zip(srow, irow)] for srow, irow in zip(scores, idx)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=str, default=None)
    ap.add_argument("--fake-embed", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--xml", type=str, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    cfg = load_config()
    ev = cfg["evidence"]
    out = ROOT / ev["dir"]

    if args.search:
        idx = EvidenceIndex(cfg, fake=args.fake_embed)
        for score, p in idx.search([args.search], ev["top_k"])[0]:
            print(f"{score:.3f} | {p['text'][:160]} | {p['url']}")
        return

    t0 = time.monotonic()
    if args.xml:
        xml_bytes = open(args.xml, "rb").read()
    else:
        blob = download(ev["medlineplus_file"], ROOT / "data" / "raw")
        if ev["medlineplus_file"].endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                name = next(n for n in z.namelist() if n.endswith(".xml"))
                xml_bytes = z.read(name)
        else:
            xml_bytes = blob
    topics = parse_topics(xml_bytes)
    passages = chunk(topics, ev["passage_sentences"], ev["passage_overlap"])
    print(f"{len(topics)} English topics -> {len(passages)} passages")
    emb = Embedder(ev["embedding_model"], fake=args.fake_embed)([p["text"] for p in passages])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "passages.jsonl", "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    np.save(out / "embeddings.npy", emb)
    backend = "numpy"
    try:
        import faiss
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(out / "index.faiss"))
        backend = f"faiss {faiss.__version__}"
    except Exception as exc:  # noqa: BLE001
        print(f"FAISS unavailable ({type(exc).__name__}); the numpy fallback gives identical results")
    meta = {"source": ev["medlineplus_file"], "topics": len(topics), "passages": len(passages),
            "embedding_model": ev["embedding_model"], "dim": int(emb.shape[1]), "backend": backend}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"built {ev['dir']} with {backend} in {(time.monotonic() - t0) / 60:.1f} min: {meta}")


if __name__ == "__main__":
    main()
