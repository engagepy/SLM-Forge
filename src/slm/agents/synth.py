"""Synth: turns human feedback into new training examples.

- sft: a teacher model writes ideal answers to new prompts in the style the human rewarded.
- preference: the teacher writes the ideal answer; the *local* model's own answer is the
  rejected side when available (on-policy pairs teach the most), else the teacher writes a
  plausible weak answer.

Everything is saved unapproved; the user reviews before any of it reaches training.
Prompts that match held-out eval prompts are dropped to prevent contamination.
"""

from sqlmodel import Session, select

from slm.agents.base import eval_prompts, event_logger, normalise, project_context
from slm.agents.provider import get_provider
from slm.db import Feedback, PreferencePair, Project, SftExample, engine

AGENT = "synth"
MAX_PER_CALL = 20

SYSTEM = """You write training data for a small language model that is being fine-tuned locally.
You are given the project's goal and examples of human feedback: prompts, the answer the human
preferred or wrote, and their critiques.

Write NEW examples that teach what the feedback rewards and avoid what it penalises:
- Vary the prompts: different topics, phrasings and difficulty within the goal's scope. Never copy
  or lightly paraphrase a feedback prompt.
- ideal_response must be exactly what a great answer looks like for this project (tone, format,
  length, correctness). The small model will imitate it, so keep it concise and unambiguous.
- weak_response (when asked for) should be a realistic mistake the small model makes, matching
  the critiques, not a strawman.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "ideal_response": {"type": "string"},
                    "weak_response": {"type": "string"},
                },
            },
        }
    },
}


def _feedback_block(project_id: int, feedback_ids: list[int] | None, limit: int = 12) -> str:
    with Session(engine()) as s:
        q = select(Feedback).where(Feedback.project_id == project_id)
        if feedback_ids:
            q = q.where(Feedback.id.in_(feedback_ids))
        rows = s.exec(q.order_by(Feedback.id.desc()).limit(limit)).all()
    lines = []
    for f in rows:
        preferred = f.edited_answer or (f.candidate_a if f.choice == "a" else f.candidate_b if f.choice == "b" else "")
        rejected = f.candidate_b if f.choice == "a" else f.candidate_a if f.choice == "b" else ""
        lines.append(
            f"- prompt: {f.prompt[:500]}\n  preferred: {preferred[:600] or '(neither was acceptable)'}\n"
            f"  rejected: {rejected[:400]}\n  critique: {f.critique or '-'}"
        )
    return "\n".join(lines) or "(no feedback yet)"


def _existing_prompts(project_id: int) -> set[str]:
    with Session(engine()) as s:
        pairs = s.exec(select(PreferencePair.prompt).where(PreferencePair.project_id == project_id)).all()
        sft = s.exec(select(SftExample.messages).where(SftExample.project_id == project_id)).all()
    out = {normalise(p) for p in pairs}
    for msgs in sft:
        out |= {normalise(m["content"]) for m in msgs if m["role"] == "user"}
    return out


def _local_answer(project: Project, prompt: str) -> str | None:
    """The served model's own answer, if the GPU is free."""
    from slm.inference.engine import EngineBusy, SamplingParams
    from slm.inference.engine import engine as infer

    if not project.current_model_path:
        return None
    messages = [{"role": "user", "content": prompt}]
    if project.system_prompt:
        messages.insert(0, {"role": "system", "content": project.system_prompt})
    try:
        text, _ = infer.generate(
            messages,
            SamplingParams(temperature=0.8, max_tokens=384),
            model_path=project.current_model_path,
            adapter_path=project.current_adapter_path,
        )
        return text.strip() or None
    except EngineBusy:
        return None


def run(
    project_id: int,
    *,
    kind: str = "sft",
    count: int = 20,
    focus: str = "",
    feedback_ids: list[int] | None = None,
    on_policy: bool = True,
) -> dict:
    if kind not in ("sft", "preference"):
        raise ValueError("kind must be sft or preference")
    ctx = project_context(project_id)
    on_event = event_logger(project_id, AGENT)
    blocked = eval_prompts(project_id) | _existing_prompts(project_id)
    feedback = _feedback_block(project_id, feedback_ids)
    provider = get_provider()

    with Session(engine()) as s:
        project = s.get(Project, project_id)
    return _generate_all(project_id, project, provider, ctx, on_event, blocked, feedback, kind, count, focus, on_policy)


def _generate_all(
    project_id, project, provider, ctx, on_event, blocked, feedback, kind, count, focus, on_policy
) -> dict:

    stats = {
        "requested": count,
        "generated": 0,
        "saved": 0,
        "dropped_contamination": 0,
        "dropped_duplicate": 0,
        "dropped_empty": 0,
        "on_policy_rejected": 0,
    }
    remaining = count
    while remaining > 0:
        n = min(remaining, MAX_PER_CALL)
        remaining -= n
        user = (
            f"Project goal: {ctx['goal']}\nSystem prompt the model runs with: {ctx['system_prompt'] or '(none)'}\n"
            f"Focus for this batch: {focus or 'whatever the feedback shows the model needs most'}\n\n"
            f"Human feedback so far:\n{feedback}\n\n"
            f"Write {n} new examples. "
            + ("Include weak_response for every example." if kind == "preference" else 'Set weak_response to "".')
        )
        on_event("message", {"text": f"Generating {n} {kind} examples (focus: {focus or 'auto'})"})
        examples = provider.json(SYSTEM, user, SCHEMA).get("examples", [])
        stats["generated"] += len(examples)

        with Session(engine()) as s:
            for ex in examples:
                prompt, ideal = ex.get("prompt", "").strip(), ex.get("ideal_response", "").strip()
                key = normalise(prompt)
                if not prompt or not ideal:
                    stats["dropped_empty"] += 1
                    continue
                if key in blocked:
                    stats["dropped_contamination" if key in eval_prompts(project_id) else "dropped_duplicate"] += 1
                    continue
                blocked.add(key)
                if kind == "sft":
                    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": ideal}]
                    if project.system_prompt:
                        msgs.insert(0, {"role": "system", "content": project.system_prompt})
                    s.add(SftExample(project_id=project_id, messages=msgs, source="synthetic", approved=False))
                else:
                    rejected = _local_answer(project, prompt) if on_policy else None
                    if rejected and normalise(rejected) != normalise(ideal):
                        stats["on_policy_rejected"] += 1
                    else:
                        rejected = ex.get("weak_response", "").strip()
                    if not rejected or normalise(rejected) == normalise(ideal):
                        stats["dropped_empty"] += 1
                        continue
                    s.add(
                        PreferencePair(
                            project_id=project_id,
                            prompt=prompt,
                            system=project.system_prompt,
                            chosen=ideal,
                            rejected=rejected,
                            source="synthetic",
                            approved=False,
                        )
                    )
                stats["saved"] += 1
            s.commit()

    on_event("message", {"text": f"Synthetic data ready for review: {stats}"})
    return stats
