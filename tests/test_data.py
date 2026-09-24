import pytest

from slm.data import clean, split
from slm.data import format as fmt


def test_instruction_mapping_with_input_and_constant_system():
    row = {"instruction": "Translate", "input": "bonjour", "output": "hello"}
    rec = fmt.map_row(
        row,
        {
            "format": "instruction",
            "prompt": "instruction",
            "input": "input",
            "response": "output",
            "system": "=Be brief.",
        },
    )
    assert rec["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Translate\n\nbonjour"},
        {"role": "assistant", "content": "hello"},
    ]


def test_sharegpt_turns_are_normalised():
    row = {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]}
    rec = fmt.map_row(row, {"format": "chat", "messages": "conversations"})
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]


def test_preference_mapping_extracts_last_assistant_turn():
    row = {
        "prompt": "2+2?",
        "chosen": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}],
        "rejected": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "5"}],
    }
    rec = fmt.map_row(row, fmt.guess_mapping(list(row)))
    assert rec == {"prompt": "2+2?", "chosen": "4", "rejected": "5"}


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        fmt.map_row({"messages": [{"role": "narrator", "content": "x"}]}, {"format": "chat"})


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (["instruction", "input", "output"], "instruction"),
        (["question", "answer"], "instruction"),
        (["messages"], "chat"),
        (["prompt", "chosen", "rejected"], "preference"),
        (["text"], "text"),
        (["foo"], "unknown"),
    ],
)
def test_guess_mapping(columns, expected):
    assert fmt.guess_mapping(columns)["format"] == expected


def _chat(q, a):
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}


def test_clean_dedupes_case_insensitively_and_counts_drops():
    records = [
        _chat("Hi", "Hello there"),
        _chat("hi", "hello there"),  # duplicate after lowercasing
        _chat("Q", ""),  # empty assistant turn
        _chat("Q2", "ok���"),  # garbled
        {"messages": [{"role": "user", "content": "no answer"}]},
    ]
    kept, report = clean.clean(records)
    assert len(kept) == 1
    assert report.dropped == {"duplicate": 1, "empty_turn": 1, "garbled": 1, "no_assistant": 1}


def test_clean_normalises_text():
    kept, _ = clean.clean([_chat("  a\r\nb\x07 ", "fine answer")])
    assert kept[0]["messages"][0]["content"] == "a\nb"


def test_clean_drops_identical_preference_pairs():
    kept, report = clean.clean([{"prompt": "p", "chosen": "same", "rejected": "same"}])
    assert not kept and report.dropped["identical_pair"] == 1


def test_split_always_leaves_a_validation_row():
    parts = split.split_records([_chat(str(i), "a") for i in range(5)])
    assert len(parts["valid"]) == 1 and len(parts["train"]) == 4 and not parts["test"]


def test_split_is_deterministic():
    recs = [_chat(str(i), "a") for i in range(50)]
    assert split.split_records(recs, seed=3) == split.split_records(recs, seed=3)


def test_token_stats():
    stats = split.token_stats([10, 20, 30, 400], max_seq_length=256)
    assert stats["max"] == 400 and stats["over_max_seq_length"] == 1 and stats["total_tokens"] == 460


RECIPE = {
    "title": "Risotto",
    "ingredients": ["200g rice", "1 onion"],
    "method": ["Fry the onion.", "Add rice and stock for 18 minutes."],
}


def test_recipe_columns_are_recognised():
    assert fmt.guess_mapping(list(RECIPE)) == {
        "format": "instruction",
        "prompt": "title",
        "response": "method",
        "input": "ingredients",
    }


def test_list_cells_become_readable_lines():
    rec = fmt.map_row(RECIPE, fmt.guess_mapping(list(RECIPE)))
    assert rec["messages"][0]["content"] == "Risotto\n\n- 200g rice\n- 1 onion"
    assert rec["messages"][1]["content"] == "- Fry the onion.\n- Add rice and stock for 18 minutes."


def test_unset_or_unknown_columns_are_errors_not_empty_strings():
    # Regression: an instruction mapping with no columns used to "map" every row to empty turns.
    with pytest.raises(ValueError, match="choose a column for 'prompt'"):
        fmt.map_row(RECIPE, {"format": "instruction"})
    with pytest.raises(ValueError, match="not in the data"):
        fmt.map_row(RECIPE, {"format": "instruction", "prompt": "title", "response": "steps"})


def test_constant_templates_insert_columns():
    rec = fmt.map_row(
        RECIPE, {"format": "instruction", "prompt": "=How do I make {title}? {unknown}", "response": "method"}
    )
    assert rec["messages"][0]["content"] == "How do I make Risotto? {unknown}"


def test_mojibake_repaired_without_touching_real_accents():
    assert clean.fix_mojibake("100g/3Â½oz, itâ€™s crÃ¨me") == "100g/3½oz, it’s crème"
    assert clean.fix_mojibake("naïve café, Ångström, 50°C") == "naïve café, Ångström, 50°C"


def test_synthetic_examples_are_fitted_to_the_sequence_length_too(session, project):
    # Regression: only Hub/uploaded datasets went through fit_to_length; synthetic examples (the
    # main data path) reached the trainer over-long and were silently truncated.
    from slm.data.pipeline import build_feedback_version
    from slm.db import SftExample

    short = [
        [{"role": "user", "content": f"Question {i}?"}, {"role": "assistant", "content": f"Answer {i}."}]
        for i in range(3)
    ]
    long = [{"role": "user", "content": "Explain"}, {"role": "assistant", "content": "word " * 3000}]
    session.add_all([SftExample(project_id=project.id, messages=m) for m in (*short, long)])
    session.commit()
    v = build_feedback_version(project.id, model_path=None, max_seq_length=256)
    assert v.n_train + v.n_valid == 3
    assert v.cleaning_report["dropped"]["too_long_for_max_seq_length"] == 1
    assert v.cleaning_report["length"]["policy"] == "drop" and v.token_stats["estimated"] is True
