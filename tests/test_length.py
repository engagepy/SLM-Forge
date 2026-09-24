import sys

import pytest

from slm.data.length import Measurer, fit_to_length, percentiles
from slm.train import runner


class WordTokenizer:
    """One token per whitespace-separated word; enough to test the policy precisely."""

    def encode(self, text):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)

    def apply_chat_template(self, messages, tokenize=True, **_):
        return " ".join(m["content"] for m in messages).split()


def chat(answer_words: int) -> dict:
    return {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "w " * answer_words}]}


def test_long_qa_examples_are_dropped_not_truncated():
    records = [chat(10), chat(50), chat(200), chat(900)]
    kept, rep = fit_to_length(records, 100, "auto", Measurer(WordTokenizer()))
    assert len(kept) == 2 and rep["policy"] == "drop" and rep["dropped"] == 2
    assert rep["too_long_percent"] == 50.0 and rep["estimated"] is False


def test_long_text_is_split_into_overlapping_windows_that_fit():
    text = " ".join(f"w{i}" for i in range(1000))
    kept, rep = fit_to_length([{"text": text}, {"text": "short one"}], 300, "auto", Measurer(WordTokenizer()))
    assert rep["policy"] == "split" and rep["split"]["examples"] == 1
    windows = [r["text"] for r in kept if r["text"] != "short one"]
    assert all(len(w.split()) <= 300 for w in windows)
    covered = {tok for w in windows for tok in w.split()}
    assert covered == set(text.split())  # nothing lost at the seams


def test_keep_policy_reports_without_changing_data():
    records = [chat(10), chat(500)]
    kept, rep = fit_to_length(records, 100, "keep", Measurer(WordTokenizer()))
    assert kept == records and rep["policy"] == "keep" and rep["too_long"] == 1


def test_everything_fits_means_no_policy_applied():
    kept, rep = fit_to_length([chat(5)], 100, "auto", Measurer(WordTokenizer()))
    assert len(kept) == 1 and rep["policy"] == "none needed" and rep["too_long"] == 0


def test_estimate_without_tokenizer_is_flagged_and_roughly_right():
    m = Measurer(None)
    n = m.lengths([{"text": "word " * 1000}])[0]  # 5,000 chars ≈ 1,390 tokens at 3.6 chars/token
    assert m.estimated and 1200 < n < 1600


def test_estimated_text_split_prefers_sentence_boundaries():
    text = "This is a sentence. " * 400
    kept, rep = fit_to_length([{"text": text}], 256, "split", Measurer(None))
    assert rep["split"]["into_windows"] >= 2
    assert all(w["text"].endswith(".") for w in kept[:-1])


def test_bad_policy_rejected():
    with pytest.raises(ValueError):
        fit_to_length([chat(1)], 10, "truncate", Measurer(WordTokenizer()))


def test_percentiles():
    assert percentiles(list(range(1, 101))) == {"p50": 51, "p90": 91, "p95": 96, "p99": 100, "max": 100}


def test_repeated_truncation_warning_is_collapsed(tmp_path):
    # The exact line MLX prints once per batch.
    warn = "[WARNING] Some sequences are longer than 2048 tokens. The longest sentence {} will be truncated to 2048."
    script = (
        "".join(f"print({warn.format(n)!r});" for n in (3099, 3055, 3271))
        + "print('Iter 1: Val loss 1.112, Val took 19.759s')"
    )
    lines = []
    runner.run_process(
        [sys.executable, "-c", script], log_path=tmp_path / "log.txt", on_line=lines.append, should_cancel=lambda: False
    )
    assert lines[0].startswith("[WARNING]") and "3099" in lines[0]
    assert lines[1].startswith("Iter 1")
    assert lines[2] == "[2 more 'sequence truncated' warnings suppressed]"
