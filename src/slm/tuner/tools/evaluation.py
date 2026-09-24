"""Evaluation and feedback: trying, scoring, comparing, AI review, and declaring the work done."""

import time
from collections import Counter

from sqlmodel import Session, select

from slm.agents.provider import get_provider
from slm.db import (
    Feedback,
    Project,
    engine,
    now,
    studio_state,
)
from slm.db import test_cases as db_test_cases
from slm.feedback import record_feedback
from slm.tuner.tools import generate as gen
from slm.tuner.tools._core import Ctx, _check_stop, _served_checkpoint_id, _spend, studio, tool
from slm.tuner.tools.scoring import _score_all
from slm.tuner.util import clip as _clip


@tool
def try_model(
    ctx: Ctx, prompts: list[str], target: str = "current", temperature: float = 0.7, max_tokens: int = 300
) -> list[dict]:
    """Ask the local model a few questions and see its answers (also shown on the canvas), to show
    the user what it sounds like. For a score, use evaluate_model. target: current (served) | base
    (untrained) | checkpoint:<id>. Use 2–4 prompts; each one loads the model onto the GPU."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        project = s.get(Project, pid)
    results = []
    for q in prompts[:6]:
        r = gen._generate(project, q, temperature, max_tokens, target)
        results.append({"prompt": q, "target": target, **r})
    with studio(pid, "samples") as st:
        st.samples = (results + list(st.samples))[:12]
    return results


@tool
def evaluate_model(ctx: Ctx, target: str = "current", reason: str = "") -> dict:
    """Score a model on the project's fixed test questions (set with update_project): each answer is
    graded 0–10 by GPT-6 against the goal and system prompt. The result is kept per checkpoint and
    shown on the canvas, so before/after and run-to-run comparisons are numbers. Run it on the base
    model before training (the score to beat), after every run, and before proposing an export.
    target: current (served) | base | checkpoint:<id>. Costs one judge call per question. Outside a
    round the user set in motion it's a proposal they confirm first; give a plain-words `reason`."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        st = studio_state(s, pid)  # first: creating the row commits, which would expire `project`
        previous = [dict(e) for e in st.evals]
        project = s.get(Project, pid)
        cases = db_test_cases(project)
    questions = [c["input"] for c in cases]
    if not cases:
        raise ValueError("No test cases yet: set them with update_project(test_cases=[...]) first.")
    if proposal := _spend(
        pid, "evaluate_model", "evaluate", f"Score the {target.split(':')[0]} model on {len(questions)} test questions",
        reason, {"questions": len(questions), "target": target}, {"target": target},
    ):  # fmt: skip
        return proposal
    items = _score_all(project, get_provider(), cases, target)
    mean = round(sum(i["score"] for i in items) / len(items), 1)
    checked = [i for i in items if i.get("exact") is not None]
    exact_rate = round(sum(1 for i in checked if i["exact"]) / len(checked), 2) if checked else None
    with Session(engine()) as s:
        project = s.get(Project, pid)
        if target == "base":
            checkpoint_id = None
        elif target.startswith("checkpoint:"):
            checkpoint_id = int(target.split(":", 1)[1])
        else:
            checkpoint_id = _served_checkpoint_id(s, project)
        record = {
            "id": f"e{int(time.time() * 1000)}",
            "target": "base" if checkpoint_id is None else f"checkpoint:{checkpoint_id}",
            "checkpoint_id": checkpoint_id,
            "mean": mean,
            "exact_rate": exact_rate,
            "items": items,
            "at": now().isoformat(),
        }
    with studio(pid, "evals") as st:
        st.evals = (previous + [record])[-60:]  # the whole history: disk is not the constraint
        st.stage = "evaluate"
    best = max(previous, key=lambda e: e["mean"], default=None)
    return {
        "target": record["target"],
        "checkpoint_id": checkpoint_id,
        "mean_score": mean,
        "exact_match_rate": exact_rate,  # over cases with an expected output; None if there are none
        "scores": [
            {"q": _clip(i["prompt"], 80), "kind": i["kind"], "score": i["score"], "why": _clip(i["reason"], 120)}
            for i in items
        ],
        "previous_best": {k: best[k] for k in ("target", "checkpoint_id", "mean")} if best else None,
        "note": "Compare with the base score and the previous best; export the best-scoring checkpoint.",
    }


@tool
def ask_user_to_compare(ctx: Ctx, prompts: list[str]) -> dict:
    """Put side-by-side A/B answers on the canvas for the *user* to judge. Only use this when the
    user has said they want to judge answers themselves; otherwise use ai_review_answers. You'll be
    told when they've finished."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        project = s.get(Project, pid)
    items = []
    for q in prompts[:10]:
        a, b = gen._two_answers(project, q, 300)
        items.append(
            {
                "id": f"c{int(time.time() * 1000)}{len(items)}",
                "prompt": q,
                "a": a,
                "b": b,
                "params_a": gen.PARAMS_A,
                "params_b": gen.PARAMS_B,
                "status": "pending",
            }
        )
    identical = sum(1 for i in items if i["a"] == i["b"])
    with studio(pid, "comparisons") as st:
        st.comparisons = [dict(c) for c in st.comparisons if c.get("status") == "pending"] + items
        st.stage = "evaluate"
    return {
        "queued": len(items),
        "identical_pairs": identical,
        "note": "Identical pairs carry no preference signal; the user can still rewrite the answer.",
    }


@tool
def feedback_summary(ctx: Ctx, limit: int = 15) -> dict:
    """The user's recent judgements and critiques, and how much training signal is ready."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        rows = s.exec(
            select(Feedback).where(Feedback.project_id == pid).order_by(Feedback.id.desc()).limit(limit)
        ).all()
    return {
        "recent": [
            {
                "prompt": _clip(f.prompt, 120),
                "choice": f.choice,
                "critique": f.critique,
                "rewrote": bool(f.edited_answer),
            }
            for f in rows
        ],
        "counts": {"judgements": len(rows)},
    }


JUDGE_SYSTEM = """You are an expert reviewer grading a small language model that is being fine-tuned
for a specific goal. For one prompt you see two candidate answers, A and B, from the model.
Decide which is better for the goal (accuracy first, then following the system prompt's style),
or "tie" if equally good, or "both_bad" if neither is acceptable. Then write the ideal answer, in
the exact style the goal and system prompt ask for, and a one-sentence critique naming the concrete
flaws. If the better answer is already ideal, repeat it verbatim as the ideal answer."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["a", "b", "tie", "both_bad"]},
        "ideal_answer": {"type": "string"},
        "critique": {"type": "string"},
    },
}


@tool
def ai_review_answers(ctx: Ctx, prompts: list[str], reason: str = "") -> dict:
    """Stand in for the human judge. For each prompt the local model writes two answers (A and B);
    GPT-6 picks the better one, writes the ideal answer and critiques the flaws. Each verdict
    becomes training signal automatically: a preference pair (better vs. worse, for DPO) and, when
    the answers were flawed, a corrected example (for SFT). Use 8–12 varied, realistic prompts
    that did NOT come from the training data. Results show on the canvas.
    This costs GPU time and API calls. Outside a round the user set in motion it's a proposal they
    confirm first; give a plain-words `reason` for the card."""
    pid = ctx.context.project_id
    if proposal := _spend(
        pid, "ai_review_answers", "review", f"Have GPT-6 review {len(prompts[:12])} answers", reason,
        {"prompts": len(prompts[:12])}, {"prompts": prompts[:12]},
    ):  # fmt: skip
        return proposal
    with Session(engine()) as s:
        project = s.get(Project, pid)
    judge = get_provider()
    verdicts = []
    for q in prompts[:12]:
        _check_stop(pid)
        a, b = gen._two_answers(project, q, 350)
        v = judge.json(
            JUDGE_SYSTEM,
            f"Goal: {project.goal}\nSystem prompt: {project.system_prompt or '(none)'}\n\nPrompt: {q}\n\n"
            f"Answer A:\n{a}\n\nAnswer B:\n{b}",
            JUDGE_SCHEMA,
        )
        choice = v.get("choice", "tie")
        chosen = a if choice == "a" else b if choice == "b" else ""
        ideal = (v.get("ideal_answer") or "").strip()
        # Only store the ideal answer as a correction when it differs from what the model said
        # (for both_bad nothing was chosen, so any ideal answer is a correction).
        edited = ideal if ideal and ideal != chosen.strip() else ""
        out = record_feedback(
            pid,
            prompt=q,
            system=project.system_prompt,
            candidate_a=a,
            candidate_b=b,
            choice=choice,
            edited_answer=edited,
            critique=v.get("critique", "") or "reviewed by AI",
            params_a=gen.PARAMS_A,
            params_b=gen.PARAMS_B,
            model_ref="ai-judge",
        )
        verdicts.append(
            {
                "id": f"ai{out['feedback_id']}",
                "prompt": q,
                "a": a,
                "b": b,
                "status": "judged",
                "judge": "ai",
                "choice": choice,
                "critique": v.get("critique", ""),
                "ideal": edited,
                "pairs": out["preference_pairs"],
                "sft": out["sft_examples"],
            }
        )
    with studio(pid, "comparisons") as st:
        st.comparisons = [dict(c) for c in st.comparisons][-20:] + verdicts
        st.stage = "refine"
    return {
        "reviewed": len(verdicts),
        "verdicts": dict(Counter(v["choice"] for v in verdicts)),
        "preference_pairs_added": sum(v["pairs"] for v in verdicts),
        "corrected_examples_added": sum(v["sft"] for v in verdicts),
        "critiques": [_clip(v["critique"], 160) for v in verdicts],
    }


@tool
def finish_project(ctx: Ctx, summary: str) -> str:
    """Declare the work done, once the model is exported (or you've decided it can't get better).
    Stops autopilot until the user wants more: they can always keep improving the model later.
    summary: two or three sentences on what was built and how good it is."""
    with studio(ctx.context.project_id) as st:
        st.completed, st.stage, st.note = True, "export", summary[:200]
    return "finished"
