"""Make examples fit the training sequence length before training, instead of letting the trainer
silently truncate them.

MLX truncates anything longer than max_seq_length, which cuts off the *end* of each example. For
question→answer data the end is the conclusion, so the model learns answers that stop mid-way.
What to do about a long example depends on the format:

- text (raw documents): split into overlapping windows that fit. Nothing is lost.
- chat / instruction / preference: drop it. Splitting an answer into two examples would teach the
  model to end its answer half-way (each piece ends with an end-of-turn token), and the second
  piece would have no question in front of it.
- keep: leave it to the trainer's truncation (not recommended; reported, not hidden).
"""

from dataclasses import dataclass

from slm.data import split as splitting

CHARS_PER_TOKEN = 3.6  # English prose with BPE tokenizers; used only when no tokenizer is available
TEMPLATE_OVERHEAD = 24  # chat-template tokens per record (role markers, BOS/EOS)
WINDOW_MARGIN = 16  # headroom for BOS/EOS when splitting text
WINDOW_OVERLAP = 64  # tokens shared between consecutive windows, so no sentence is lost at a seam

POLICIES = ("auto", "drop", "split", "keep")


@dataclass
class Measurer:
    """Token lengths as the trainer will see them; estimated from characters if there's no tokenizer yet."""

    tokenizer: object | None = None

    @property
    def estimated(self) -> bool:
        return self.tokenizer is None

    def lengths(self, records: list[dict]) -> list[int]:
        if self.tokenizer is not None:
            return splitting.token_lengths(records, self.tokenizer)
        return [self._estimate(r) for r in records]

    @staticmethod
    def _estimate(rec: dict) -> int:
        if "messages" in rec:
            chars = sum(len(m["content"]) for m in rec["messages"])
            return int(chars / CHARS_PER_TOKEN) + TEMPLATE_OVERHEAD * len(rec["messages"]) // 2
        if "text" in rec:
            return int(len(rec["text"]) / CHARS_PER_TOKEN) + 2
        longest = max(len(rec.get("chosen", "")), len(rec.get("rejected", "")))
        return int((len(rec.get("prompt", "")) + longest) / CHARS_PER_TOKEN) + TEMPLATE_OVERHEAD


def percentiles(lengths: list[int]) -> dict:
    if not lengths:
        return {}
    s = sorted(lengths)

    def at(p: float) -> int:
        return s[min(len(s) - 1, int(p * len(s)))]

    return {"p50": at(0.5), "p90": at(0.9), "p95": at(0.95), "p99": at(0.99), "max": s[-1]}


def _windows_tokens(text: str, tokenizer, size: int) -> list[str]:
    ids = tokenizer.encode(text)
    stride = max(1, size - WINDOW_OVERLAP)
    return [tokenizer.decode(ids[i : i + size]) for i in range(0, max(1, len(ids) - WINDOW_OVERLAP), stride)]


def _windows_chars(text: str, size_tokens: int) -> list[str]:
    """Character windows that prefer to end at a paragraph or sentence boundary."""
    size = int(size_tokens * CHARS_PER_TOKEN)
    overlap = int(WINDOW_OVERLAP * CHARS_PER_TOKEN)
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            tail = text[start + int(size * 0.8) : end]
            for sep in ("\n\n", ". ", "\n"):
                cut = tail.rfind(sep)
                if cut != -1:
                    end = start + int(size * 0.8) + cut + len(sep)
                    break
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [w for w in out if w]


def fit_to_length(records: list[dict], max_seq_length: int, policy: str, measurer: Measurer) -> tuple[list[dict], dict]:
    """Apply the length policy. Returns the records to keep and a report of what happened."""
    if policy not in POLICIES:
        raise ValueError(f"long_examples must be one of {POLICIES}")
    lengths = measurer.lengths(records)
    over = [i for i, n in enumerate(lengths) if n > max_seq_length]
    total_tokens = sum(lengths) or 1
    report = {
        "max_seq_length": max_seq_length,
        "estimated": measurer.estimated,
        "lengths": percentiles(lengths),
        "too_long": len(over),
        "too_long_percent": round(100 * len(over) / len(records), 1) if records else 0.0,
        "tokens_over_limit_percent": round(100 * sum(max(0, n - max_seq_length) for n in lengths) / total_tokens, 1),
    }
    if not over:
        report["policy"] = "none needed"
        return records, report

    is_text = all("text" in r for r in records)
    effective = ("split" if is_text else "drop") if policy == "auto" else policy
    report["policy"] = effective
    if effective == "keep":
        report["note"] = "Long examples left in; the trainer will cut off their endings."
        return records, report

    over_set = set(over)
    kept, split_from, windows = [], 0, 0
    for i, rec in enumerate(records):
        if i not in over_set:
            kept.append(rec)
        elif effective == "split" and "text" in rec:
            size = max_seq_length - WINDOW_MARGIN
            pieces = (
                _windows_tokens(rec["text"], measurer.tokenizer, size)
                if measurer.tokenizer is not None
                else _windows_chars(rec["text"], size)
            )
            kept.extend({"text": p} for p in pieces)
            split_from += 1
            windows += len(pieces)
        # else: dropped (chat/instruction/preference can't be split without teaching broken answers)
    if split_from:
        report["split"] = {"examples": split_from, "into_windows": windows}
    report["dropped"] = len(over) - split_from
    return kept, report
