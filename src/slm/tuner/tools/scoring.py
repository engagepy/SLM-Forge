"""Scoring answers against the test set: exact match first, then the judge."""

from slm.db import (
    Project,
)
from slm.tuner.tools import generate as gen
from slm.tuner.tools._core import _check_stop

SCORE_SYSTEM = """You grade one answer from a small language model that is being fine-tuned for a
specific goal. Score it 0–10 for how well it serves that goal: correctness first (a confident wrong
fact caps the score at 3), then whether it follows the system prompt's voice, format and length.
10 is an answer the user would be delighted by; 5 is usable but flawed; 0 is wrong or off-topic.
Give a one-sentence reason naming the concrete strength or flaw."""

SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 10}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
}


def _canon(text: str) -> tuple[str, bool]:
    """A comparable form of an output: canonical JSON when it parses, else trimmed lowercase text."""
    import json as _json

    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return _json.dumps(_json.loads(t), sort_keys=True, separators=(",", ":")), True
    except ValueError:
        return " ".join(t.lower().split()), False


def _score_all(project: Project, judge, cases: list[dict], target: str) -> list[dict]:
    """Exact match first (an expected output is the ground truth), the judge only where there is
    no expected output or the answer differs from it and may still deserve partial credit."""
    items = []
    for case in cases:
        _check_stop(project.id)
        q, expected = case["input"], case.get("expected")
        answer = gen._generate(project, q, 0.3, 300, target)["text"]
        item = {"prompt": q, "kind": case.get("kind", "on-goal"), "answer": answer, "exact": None}
        if expected:
            got, got_json = _canon(answer)
            want, want_json = _canon(expected)
            item["exact"] = got == want
            if item["exact"]:
                item |= {"score": 10, "reason": "matches the expected output"}
            elif want_json and not got_json:
                item |= {"score": 0, "reason": "not valid JSON, and JSON was expected"}
            else:
                v = judge.json(
                    SCORE_SYSTEM + "\nYou are also given the expected output: score by how close the answer is to it.",
                    f"Goal: {project.goal}\nSystem prompt: {project.system_prompt or '(none)'}\n\n"
                    f"Input: {q}\n\nExpected output:\n{expected}\n\nAnswer:\n{answer}",
                    SCORE_SCHEMA,
                )
                item |= {"score": max(0, min(9, int(v.get("score", 0)))), "reason": (v.get("reason") or "")[:300]}
        else:
            v = judge.json(
                SCORE_SYSTEM,
                f"Goal: {project.goal}\nSystem prompt: {project.system_prompt or '(none)'}\n\n"
                f"Question: {q}\n\nAnswer:\n{answer}",
                SCORE_SCHEMA,
            )
            item |= {"score": max(0, min(10, int(v.get("score", 0)))), "reason": (v.get("reason") or "")[:300]}
        items.append(item)
    return items
