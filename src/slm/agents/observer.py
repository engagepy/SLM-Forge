"""Observer: watches feedback and training results and decides what should happen next.

Deterministic rules handle the obvious triggers (enough preference pairs → DPO round). The LLM
reads the critiques and decides whether, and on what, to generate synthetic data. Every
decision is a proposal with its reasoning; nothing runs without approval.
"""

from collections import Counter

from sqlmodel import Session, select

from slm.agents.base import create_proposal, event_logger, pending_actions, project_context
from slm.agents.provider import get_provider
from slm.db import Checkpoint, Feedback, Job, Metric, PreferencePair, SftExample, engine

AGENT = "observer"
DPO_MIN_PAIRS = 8
SFT_MIN_EXAMPLES = 10

SYSTEM = """You are the Observer in a local fine-tuning loop for a small language model. Humans
compare the model's answers, pick the better one, edit ideal answers and write critiques. You read
that feedback plus training metrics and decide what should happen next.

Decide:
- assessment: 2-4 sentences on how the model is doing and its main weaknesses, grounded in the
  critiques and choices (quote short phrases). Mention overfitting if validation loss collapses
  to near zero or rises while train loss falls.
- synthetic: whether to generate synthetic training data now. Only generate when the feedback
  shows a clear, teachable pattern. Use kind "preference" when the model is roughly right but its
  style or quality varies (DPO sharpens that); use "sft" when it lacks knowledge or a format
  entirely. count between 10 and 60. focus: one sentence on exactly what to target.
- notes: short concrete suggestions for the human (sampling settings, data to add, hyperparameters)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string"},
        "synthetic": {
            "type": "object",
            "properties": {
                "should_generate": {"type": "boolean"},
                "kind": {"type": "string", "enum": ["sft", "preference"]},
                "count": {"type": "integer"},
                "focus": {"type": "string"},
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
}


def gather_state(project_id: int) -> dict:
    with Session(engine()) as s:
        fresh = s.exec(
            select(Feedback).where(Feedback.project_id == project_id, Feedback.observed == False)  # noqa: E712
        ).all()
        pairs_ready = s.exec(
            select(PreferencePair).where(
                PreferencePair.project_id == project_id,
                PreferencePair.approved == True,  # noqa: E712
                PreferencePair.used_in_job_id == None,  # noqa: E711
            )
        ).all()
        pending_review = len(
            s.exec(
                select(PreferencePair.id).where(
                    PreferencePair.project_id == project_id, PreferencePair.approved == False
                )  # noqa: E712
            ).all()
        ) + len(
            s.exec(select(SftExample.id).where(SftExample.project_id == project_id, SftExample.approved == False)).all()  # noqa: E712
        )
        sft_ready = s.exec(
            select(SftExample).where(
                SftExample.project_id == project_id,
                SftExample.approved == True,  # noqa: E712
                SftExample.used_in_job_id == None,  # noqa: E711
            )
        ).all()
        ckpts = s.exec(select(Checkpoint).where(Checkpoint.project_id == project_id).order_by(Checkpoint.id)).all()
        last_train_job = s.exec(
            select(Job)
            .where(Job.project_id == project_id, Job.kind.in_(["sft", "dpo"]), Job.status == "succeeded")
            .order_by(Job.id.desc())
        ).first()
        val_curve = []
        if last_train_job:
            val_curve = [
                round(m.values["loss"], 4)
                for m in s.exec(
                    select(Metric)
                    .where(Metric.job_id == last_train_job.id, Metric.split == "val")
                    .order_by(Metric.iteration)
                ).all()
            ]
    return {
        "fresh_feedback": fresh,
        "choices": Counter(f.choice for f in fresh),
        "pairs_ready": len(pairs_ready),
        "sft_ready": len(sft_ready),
        "pending_review": pending_review,
        "checkpoints": [{"kind": c.kind, "metrics": c.metrics} for c in ckpts],
        "last_run": {
            "kind": last_train_job.kind,
            "val_curve": val_curve,
            "warnings": (last_train_job.result or {}).get("warnings", []),
        }
        if last_train_job
        else None,
    }


def _rule_proposals(project_id: int, st: dict, pending: set[str]) -> list[str]:
    made = []
    if st["pairs_ready"] >= DPO_MIN_PAIRS and "run_dpo" not in pending:
        create_proposal(
            project_id,
            AGENT,
            "run_dpo",
            f"Run a DPO round on {st['pairs_ready']} preference pairs",
            f"{st['pairs_ready']} approved pairs haven't been trained on yet (threshold {DPO_MIN_PAIRS}). "
            "A DPO round moves the model toward the answers humans preferred.",
            {"preset": "safe"},
        )
        made.append("run_dpo")
    if st["sft_ready"] >= SFT_MIN_EXAMPLES and "run_sft" not in pending:
        create_proposal(
            project_id,
            AGENT,
            "run_sft",
            f"Fine-tune on {st['sft_ready']} corrected/synthetic examples",
            "Edited answers and approved synthetic examples are ready. SFT teaches the model to produce them directly.",
            {"source": "feedback", "preset": "safe"},
        )
        made.append("run_sft")
    return made


def run(project_id: int, *, use_llm: bool = True) -> dict:
    ctx = project_context(project_id)
    on_event = event_logger(project_id, AGENT)
    st = gather_state(project_id)
    pending = pending_actions(project_id)
    made = _rule_proposals(project_id, st, pending)
    out: dict = {"rule_proposals": made, "assessment": "", "notes": []}

    curve = (st["last_run"] or {}).get("val_curve") or []
    for w in (st["last_run"] or {}).get("warnings", []):
        out["notes"].append(w["message"])
        on_event("message", {"text": w["message"], "warning": w["code"]})

    fresh = st["fresh_feedback"]
    if use_llm and fresh:
        lines = []
        for f in fresh[:40]:
            lines.append(
                f"- [{f.choice}] prompt: {f.prompt[:300]!r}"
                + (f" | critique: {f.critique[:300]!r}" if f.critique else "")
                + (" | human wrote an edited answer" if f.edited_answer else "")
            )
        user = (
            f"Goal: {ctx['goal']}\nBase model: {ctx['base_model']}\n"
            f"Training history: {st['checkpoints'] or 'none yet'}\n"
            f"Last run validation loss curve: {curve or 'n/a'}\n"
            f"Choice counts in new feedback: {dict(st['choices'])}\n"
            f"Preference pairs ready for DPO: {st['pairs_ready']}; SFT examples ready: {st['sft_ready']}; "
            f"awaiting human review: {st['pending_review']}\n\n"
            f"New feedback ({len(fresh)} items):\n" + "\n".join(lines)
        )
        try:
            decision = get_provider().json(SYSTEM, user, SCHEMA)
            out["assessment"] = decision.get("assessment", "")
            out["notes"] += decision.get("notes", [])
            on_event("message", {"text": out["assessment"], "notes": decision.get("notes", [])})
            syn = decision.get("synthetic") or {}
            if syn.get("should_generate") and "generate_synthetic" not in pending:
                count = max(10, min(int(syn.get("count", 20)), 60))
                create_proposal(
                    project_id,
                    AGENT,
                    "generate_synthetic",
                    f"Generate {count} synthetic {syn.get('kind', 'sft')} examples",
                    syn.get("focus", ""),
                    {
                        "kind": syn.get("kind", "sft"),
                        "count": count,
                        "focus": syn.get("focus", ""),
                        "feedback_ids": [f.id for f in fresh[:40]],
                    },
                )
                made.append("generate_synthetic")
        except Exception as e:
            on_event("error", {"text": f"LLM assessment failed: {e}"})
            out["notes"].append(f"LLM assessment unavailable: {e}")

    with Session(engine()) as s:
        for f in fresh:
            f.observed = True
            s.add(f)
        s.commit()
    out["observed"] = len(fresh)
    return out
