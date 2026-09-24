"""Cleaning rules applied to mapped records. Every drop is counted in a report."""

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REPLACEMENT_RUN = re.compile("�{3,}")


@dataclass
class CleaningRules:
    min_chars: int = 2  # per response / text
    max_chars: int = 20000  # whole record
    dedupe: bool = True
    drop_garbled: bool = True  # runs of U+FFFD, the mark of broken decoding
    require_assistant: bool = True
    drop_identical_pairs: bool = True  # preference rows where chosen == rejected


@dataclass
class CleaningReport:
    input_rows: int = 0
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {"input_rows": self.input_rows, "kept": self.kept, "dropped": dict(self.dropped)}


def normalise_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = _CONTROL.sub("", s)
    s = s.replace("\r\n", "\n")
    return s.strip()


def _texts(rec: dict) -> list[str]:
    if "messages" in rec:
        return [m["content"] for m in rec["messages"]]
    if "text" in rec:
        return [rec["text"]]
    return [rec.get("prompt", ""), rec.get("chosen", ""), rec.get("rejected", "")]


def _normalise_record(rec: dict) -> dict:
    if "messages" in rec:
        return {"messages": [{**m, "content": normalise_text(m["content"])} for m in rec["messages"]]}
    return {k: normalise_text(v) if isinstance(v, str) else v for k, v in rec.items()}


def _reject_reason(rec: dict, rules: CleaningRules) -> str | None:
    texts = _texts(rec)
    if rules.drop_garbled and any(_REPLACEMENT_RUN.search(t) for t in texts):
        return "garbled"
    if sum(len(t) for t in texts) > rules.max_chars:
        return "too_long"
    if "messages" in rec:
        turns = rec["messages"]
        if rules.require_assistant and not any(m["role"] == "assistant" for m in turns):
            return "no_assistant"
        if any(not m["content"] for m in turns if m["role"] != "system"):
            return "empty_turn"
        answers = [m["content"] for m in turns if m["role"] == "assistant"]
        if answers and min(len(a) for a in answers) < rules.min_chars:
            return "too_short"
    elif "text" in rec:
        if len(rec["text"]) < rules.min_chars:
            return "too_short"
    else:
        if not rec.get("prompt") or not rec.get("chosen") or not rec.get("rejected"):
            return "empty_field"
        if rules.drop_identical_pairs and rec["chosen"] == rec["rejected"]:
            return "identical_pair"
    return None


def fingerprint(rec: dict) -> str:
    joined = "\x1e".join(t.lower() for t in _texts(rec))
    return hashlib.sha1(joined.encode()).hexdigest()


def clean(records: list[dict], rules: CleaningRules | None = None) -> tuple[list[dict], CleaningReport]:
    rules = rules or CleaningRules()
    report = CleaningReport(input_rows=len(records))
    seen: set[str] = set()
    kept = []
    for rec in records:
        rec = _normalise_record(rec)
        if reason := _reject_reason(rec, rules):
            report.dropped[reason] += 1
            continue
        if rules.dedupe:
            fp = fingerprint(rec)
            if fp in seen:
                report.dropped["duplicate"] += 1
                continue
            seen.add(fp)
        kept.append(rec)
    report.kept = len(kept)
    return kept, report
