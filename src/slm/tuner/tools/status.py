"""Where things stand: the project, this Mac, and every project on it."""

from sqlmodel import Session, select

from slm import hardware
from slm.db import (
    Checkpoint,
    Dataset,
    DatasetVersion,
    Feedback,
    Job,
    PreferencePair,
    Project,
    SftExample,
    awaiting_review,
    count,
    engine,
    ready,
    studio_state,
)
from slm.db import test_cases as db_test_cases
from slm.events import canvas_changed
from slm.models import manage
from slm.sessions import overview, resume_project, stop_project
from slm.tuner.tools._core import STAGES, Ctx, _job_brief, _served_checkpoint_id, _version_brief, studio, tool


def _machine_guide(hw) -> dict:
    """What this Mac can train comfortably, so plans are sized to it rather than to a textbook."""
    budget = hw.budget_gb
    tiers = [(0.6, "0.5B"), (1.6, "1.5B"), (3.5, "3B"), (8.0, "7B")]
    largest = hardware.max_params_for_budget(budget, 4) / 1e9
    comfortable = next((label for cap, label in reversed(tiers) if cap <= largest * 0.6), "0.5B")
    return {
        "ml_budget_gb": round(budget, 1),
        "largest_trainable_4bit": f"{largest:.1f}B",
        "comfortable_tier": comfortable,
        "note": "Above the comfortable tier, use preset safe, batch 1–2 and sequence length ≤ 512; "
        "the memory estimate underestimates 3B+ by ~30%.",
    }


@tool
def get_status(ctx: Ctx) -> dict:
    """Everything about the project right now: hardware, goal, base model, datasets, prepared
    versions, running and recent jobs, trained checkpoints (with metrics and warnings), feedback
    counts and exports. Call this first in every new turn if you're unsure where things stand."""
    pid = ctx.context.project_id
    hw = hardware.detect()
    with Session(engine()) as s:
        p = s.get(Project, pid)
        st = studio_state(s, pid)
        datasets = s.exec(select(Dataset).where(Dataset.project_id == pid)).all()
        versions = s.exec(select(DatasetVersion).where(DatasetVersion.project_id == pid)).all()
        jobs = s.exec(select(Job).where(Job.project_id == pid).order_by(Job.id.desc()).limit(8)).all()
        ckpts = s.exec(select(Checkpoint).where(Checkpoint.project_id == pid).order_by(Checkpoint.id)).all()
        served_id = _served_checkpoint_id(s, p)
        job_results = {j.id: j.result for j in s.exec(select(Job).where(Job.id.in_([c.job_id for c in ckpts]))).all()}

        feedback = {
            "judgements": count(s, Feedback, Feedback.project_id == pid),
            "preference_pairs_ready": count(
                s, PreferencePair, PreferencePair.project_id == pid, *ready(PreferencePair)
            ),
            "sft_examples_ready": count(s, SftExample, SftExample.project_id == pid, *ready(SftExample)),
            "synthetic_awaiting_review": sum(
                count(s, m, m.project_id == pid, *awaiting_review(m)) for m in (PreferencePair, SftExample)
            ),
            "comparisons_waiting_for_user": sum(1 for c in st.comparisons if c.get("status") == "pending"),
        }
        exports = s.exec(
            select(Job).where(Job.project_id == pid, Job.kind == "export", Job.status == "succeeded")
        ).all()
        return {
            "hardware": {
                "chip": hw.chip,
                "memory_gb": hw.total_memory_gb,
                "ml_budget_gb": round(hw.budget_gb, 1),
                "largest_trainable_4bit_params_b": round(hardware.max_params_for_budget(hw.budget_gb, 4) / 1e9, 1),
            },
            "project": {
                "name": p.name,
                "goal": p.goal,
                "system_prompt": p.system_prompt,
                "plan": p.plan,
                "test_cases": db_test_cases(p),
            },
            "this_mac": _machine_guide(hw),
            "evaluations": [
                {k: e.get(k) for k in ("id", "target", "checkpoint_id", "mean", "at")} for e in st.evals[-6:]
            ],
            "models_already_on_this_mac": [
                {k: m[k] for k in ("repo_id", "params_b", "bits", "fit")} for m in manage.local_models()
            ],
            "canvas_stage": st.stage,
            "waiting_for_user_to_confirm": (st.pending_action or {}).get("title"),
            "base_model": {
                "repo_id": p.base_model,
                "downloaded": bool(p.base_model and manage.local_path_for(p.base_model)),
                "serving": "trained adapter"
                if p.current_adapter_path
                else ("fused model" if p.current_model_path and ckpts else "base model"),
            },
            "datasets": [
                {
                    "dataset_id": d.id,
                    "name": d.name,
                    "source": d.source_ref or d.source,
                    "rows": d.n_rows,
                    "columns": d.columns,
                    "license": d.license,
                }
                for d in datasets
            ],
            "prepared_versions": [_version_brief(v) for v in versions],
            "recent_jobs": [_job_brief(j) for j in jobs],
            "checkpoints": [
                {
                    "checkpoint_id": c.id,
                    "kind": c.kind,
                    "job_id": c.job_id,
                    "from_base": c.parent_id is None and i > 0,
                    "metrics": c.metrics,
                    "warnings": [w["code"] for w in (job_results.get(c.job_id) or {}).get("warnings", [])],
                    "serving": c.id == served_id,
                }
                for i, c in enumerate(ckpts)
            ],
            "feedback": feedback,
            "exports": [
                {
                    "path": j.result.get("path"),
                    "size_gb": j.result.get("size_gb"),
                    "min_ram_gb": j.result.get("min_ram_gb"),
                    "system_prompt_built_in": bool(j.result.get("system_prompt_built_in")),
                }
                for j in exports
            ],
        }


@tool
def set_stage(ctx: Ctx, stage: str, note: str) -> str:
    """Move the live canvas to a pipeline stage and show one short line about what's happening.
    stage: goal | data | model | train | evaluate | refine | export. Call it whenever the work moves on."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    with studio(ctx.context.project_id) as st:
        st.stage, st.note = stage, note[:200]
    return f"canvas → {stage}"


@tool
def update_project(
    ctx: Ctx,
    name: str | None = None,
    goal: str | None = None,
    system_prompt: str | None = None,
    plan: dict | None = None,
    test_cases: list[dict | str] | None = None,
) -> dict:
    """Record the project's name, goal (one or two sentences), the system prompt the model will be
    trained and used with, your plan, and the test set. Set the first three as soon as you
    understand the goal; write the plan before any data.
    plan: {task_type: persona|qa|extraction|classification|other, output_format, model_tier,
    data_target (how many examples and why), eval_design (how many cases, scored how),
    stop_rule (what score or condition ends the work)}. It's shown to the user, who can steer it.
    test_cases: [{input, expected?, kind?}]. input is what a user would type. expected is the exact
    correct output when one exists (JSON, a label, a number): those are scored by exact match
    first, so give them for every deterministic task. kind: on-goal | should-not (the right answer
    is the empty/negative result) | edge. Fixed once training starts: evaluate_model scores every
    checkpoint on the same set. Replaces the whole set."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        p = s.get(Project, pid)
        if name:
            p.name = name[:60]
        if goal:
            p.goal = goal
        if system_prompt is not None:
            p.system_prompt = system_prompt
        if plan is not None:
            p.plan = {k: str(v)[:400] for k, v in plan.items() if v is not None}
        if test_cases is not None:
            cases = []
            for c in test_cases[:60]:
                c = {"input": c} if isinstance(c, str) else dict(c)
                if str(c.get("input", "")).strip():
                    cases.append(
                        {
                            "input": str(c["input"]).strip()[:600],
                            "expected": (str(c["expected"]).strip()[:2000] if c.get("expected") else None),
                            "kind": c.get("kind") if c.get("kind") in ("on-goal", "should-not", "edge") else "on-goal",
                        }
                    )
            p.test_questions = cases
        s.add(p)
        s.commit()
        s.refresh(p)
    canvas_changed(pid)
    return {
        "name": p.name,
        "goal": p.goal,
        "system_prompt": p.system_prompt,
        "plan": p.plan,
        "test_cases": db_test_cases(p),
    }


@tool
def machine_overview(ctx: Ctx) -> dict:
    """What this Mac is doing across ALL projects: the training run using the GPU (project,
    progress, minutes left), the queue of training runs waiting, and every project's state.
    Only one model trains at a time on this Mac; others queue. Check this before starting a
    training run, and tell the user if theirs will wait behind another project (and roughly how long)."""
    data = overview()
    data["this_project_id"] = ctx.context.project_id
    return data


@tool
def manage_project(ctx: Ctx, project_id: int, action: str) -> dict:
    """Pause, resume or stop ANOTHER project (or this one). action: pause (stop its autopilot; a
    running job finishes), stop (pause and cancel its queued/running jobs, freeing the GPU), resume
    (turn its autopilot back on). Only act on another project when the user asks you to, e.g. to
    free the GPU for this one; explain what will happen first (a stopped training run is lost)."""
    if action == "resume":
        return resume_project(project_id)
    if action in ("pause", "stop"):
        return stop_project(project_id, cancel_jobs=action == "stop")
    raise ValueError("action must be pause, stop or resume")
