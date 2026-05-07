from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
except Exception:
    _nltk_sent_tokenize = None

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "datasets" / "splits"

SAMPLE_BOOKS = 2000
TARGET_BENCHMARK = 2000
TARGET_DEV = 500
TARGET_SMOKE = 200
MIN_CHAR = 20
MAX_CHAR = 500
MIN_WORDS = 5
MAX_WORDS = 60
SEED = 42


def sent_tokenize(text: str) -> list[str]:
    if _nltk_sent_tokenize is not None:
        return _nltk_sent_tokenize(text)
    import re
    # Deterministic fallback when nltk is unavailable in the standalone environment.
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _example(split: str, idx: int, text: str) -> dict:
    import hashlib
    return {
        "example_id": f"{split}_{idx}",
        "text": text,
        "source_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
    }


def _save_jsonl(name: str, texts: list[str]) -> Path:
    path = SPLITS_DIR / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for idx, text in enumerate(texts):
            f.write(json.dumps(_example(name, idx, text), ensure_ascii=False) + "\n")
    return path


def main() -> None:
    random.seed(SEED)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("bookcorpusopen", "plain_text", split="train")
    sampled_books = random.sample(list(range(len(ds))), SAMPLE_BOOKS)

    sentences: list[str] = []
    for book_idx in sampled_books:
        text = ds[book_idx]["text"]
        for block in text.split("\n"):
            block = block.strip()
            if not block:
                continue
            for sent in sent_tokenize(block):
                sent = sent.strip()
                if not (MIN_CHAR < len(sent) < MAX_CHAR):
                    continue
                wc = len(sent.split())
                if not (MIN_WORDS <= wc <= MAX_WORDS):
                    continue
                sentences.append(sent)

    random.shuffle(sentences)
    needed = TARGET_BENCHMARK + TARGET_DEV + TARGET_SMOKE
    sampled = sentences[:needed]
    if len(sampled) < needed:
        raise RuntimeError(f"Not enough sentences after filtering: {len(sampled)} < {needed}")

    smoke = sampled[:TARGET_SMOKE]
    dev = sampled[TARGET_SMOKE:TARGET_SMOKE + TARGET_DEV]
    benchmark = sampled[TARGET_SMOKE + TARGET_DEV:needed]

    smoke_path = _save_jsonl("smoke", smoke)
    dev_path = _save_jsonl("dev", dev)
    benchmark_path = _save_jsonl("benchmark", benchmark)

    print(json.dumps({
        "smoke": {"count": len(smoke), "path": str(smoke_path)},
        "dev": {"count": len(dev), "path": str(dev_path)},
        "benchmark": {"count": len(benchmark), "path": str(benchmark_path)},
    }, indent=2))


if __name__ == "__main__":
    main()
