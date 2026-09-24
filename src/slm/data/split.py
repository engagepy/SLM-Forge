"""Split cleaned records into train/valid/test JSONL and measure token lengths."""

import json
import random
from pathlib import Path


def split_records(
    records: list[dict], *, valid_frac: float = 0.05, test_frac: float = 0.05, seed: int = 0
) -> dict[str, list[dict]]:
    rows = records[:]
    random.Random(seed).shuffle(rows)
    n = len(rows)
    # mlx_lm needs a non-empty validation set; keep at least 1 when there is data to spare.
    n_valid = max(1, round(n * valid_frac)) if n >= 3 else 0
    n_test = max(1, round(n * test_frac)) if n >= 10 and test_frac > 0 else 0
    return {
        "valid": rows[:n_valid],
        "test": rows[n_valid : n_valid + n_test],
        "train": rows[n_valid + n_test :],
    }


def write_splits(splits: dict[str, list[dict]], dest: Path) -> dict[str, int]:
    dest.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in splits.items():
        if not rows:
            continue
        with open(dest / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[name] = len(rows)
    return counts


def token_lengths(records: list[dict], tokenizer) -> list[int]:
    """Token count per record as the trainer will see it (chat template applied)."""
    out = []
    for r in records:
        if "messages" in r:
            ids = tokenizer.apply_chat_template(r["messages"], tokenize=True)
        elif "text" in r:
            ids = tokenizer.encode(r["text"])
        else:  # preference: the longer of the two full conversations
            base = [{"role": "user", "content": r["prompt"]}]
            if r.get("system"):
                base.insert(0, {"role": "system", "content": r["system"]})
            ids = max(
                (
                    tokenizer.apply_chat_template(base + [{"role": "assistant", "content": r[k]}], tokenize=True)
                    for k in ("chosen", "rejected")
                ),
                key=len,
            )
        if isinstance(ids, dict):  # some tokenizers return BatchEncoding
            ids = ids["input_ids"]
        out.append(len(ids))
    return out


def token_stats(lengths: list[int], max_seq_length: int) -> dict:
    if not lengths:
        return {}
    s = sorted(lengths)

    def pct(p: float) -> int:
        return s[min(len(s) - 1, int(p * len(s)))]

    return {
        "count": len(s),
        "min": s[0],
        "mean": round(sum(s) / len(s), 1),
        "p50": pct(0.5),
        "p95": pct(0.95),
        "max": s[-1],
        "total_tokens": sum(s),
        "over_max_seq_length": sum(1 for x in s if x > max_seq_length),
        "max_seq_length": max_seq_length,
    }
