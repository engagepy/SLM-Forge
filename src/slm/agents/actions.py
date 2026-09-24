"""Agent job handlers, and execution of approved proposals.

Approving a proposal is the human gate. This module turns an approved proposal into the
job that carries it out and links the two.
"""

from sqlmodel import Session

from slm.agents import observer, prep, scout, synth
from slm.agents.base import set_proposal_status
from slm.db import Project, Proposal, engine
from slm.models import manage
from slm.train.config import preset
from slm.train.worker import JobContext, worker

# ── agent jobs ──────────────────────────────────────────────────────────────


@worker.register("agent_scout")
def scout_job(ctx: JobContext) -> None:
    result = scout.run(ctx.project_id, ctx.config.get("request", ""))
    ctx.result = {"summary": result.text, "steps": result.steps, "tool_calls": len(result.tool_calls)}


@worker.register("agent_prep")
def prep_job(ctx: JobContext) -> None:
    prop = prep.run(ctx.config["dataset_id"], use_llm=ctx.config.get("use_llm", True))
    ctx.result = {"proposal_id": prop.id}


@worker.register("agent_observer")
def observer_job(ctx: JobContext) -> None:
    ctx.result = observer.run(ctx.project_id, use_llm=ctx.config.get("use_llm", True))


@worker.register("synthesize")
def synth_job(ctx: JobContext) -> None:
    c = ctx.config
    ctx.result = synth.run(
        ctx.project_id,
        kind=c.get("kind", "sft"),
        count=int(c.get("count", 20)),
        focus=c.get("focus", ""),
        feedback_ids=c.get("feedback_ids"),
        on_policy=c.get("on_policy", True),
    )
    if pid := c.get("proposal_id"):
        set_proposal_status(pid, "executed", {"synthesis": ctx.result})


# ── proposal execution ──────────────────────────────────────────────────────


def _train_config(project_id: int, mode: str, preset_name: str, overrides: dict | None) -> dict:
    with Session(engine()) as s:
        project = s.get(Project, project_id)
        model_path = project.current_model_path or manage.local_path_for(project.base_model or "")
    if not model_path:
        raise ValueError("Download the project's base model first")
    cfg = preset(preset_name, manage.read_shape(model_path), mode)
    return cfg.model_dump() | (overrides or {})


def execute(proposal_id: int, overrides: dict | None = None) -> dict:
    """Run an approved proposal. `overrides` lets the UI edit the payload at approval time."""
    with Session(engine()) as s:
        prop = s.get(Proposal, proposal_id)
        if prop is None:
            raise KeyError(proposal_id)
        if prop.status not in ("pending", "failed"):
            raise ValueError(f"proposal is already {prop.status}")
    payload = prop.payload | (overrides or {})
    pid = prop.project_id
    set_proposal_status(proposal_id, "approved")

    try:
        match prop.action:
            case "import_dataset":
                job = worker.submit(
                    "import_dataset",
                    {k: payload.get(k) for k in ("repo_id", "config", "split", "max_rows", "license")},
                    pid,
                )
            case "prepare_dataset":
                job = worker.submit("prepare_dataset", {k: v for k, v in payload.items() if k != "preview"}, pid)
            case "run_sft":
                from slm.data.pipeline import build_feedback_version

                vid = payload.get("dataset_version_id")
                if vid is None:
                    with Session(engine()) as s:
                        project = s.get(Project, pid)
                        model_path = project.current_model_path
                    vid = build_feedback_version(pid, model_path=model_path).id
                train = _train_config(pid, "sft", payload.get("preset", "safe"), payload.get("train"))
                job = worker.submit("sft", {"train": train, "dataset_version_id": vid}, pid)
            case "run_dpo":
                train = _train_config(pid, "dpo", payload.get("preset", "safe"), payload.get("train"))
                job = worker.submit("dpo", {"train": train}, pid)
            case "generate_synthetic":
                job = worker.submit("synthesize", payload | {"proposal_id": proposal_id}, pid)
            case "acquire_manually":
                # Nothing to run: the user uploads the file against this proposal.
                return set_proposal_status(proposal_id, "approved", {"awaiting": "upload"}).model_dump(mode="json")
            case _:
                raise ValueError(f"unknown action {prop.action!r}")
    except Exception as e:
        set_proposal_status(proposal_id, "failed", {"error": str(e)})
        raise
    status = "approved" if prop.action == "generate_synthetic" else "executed"
    return set_proposal_status(proposal_id, status, {"job_id": job.id}).model_dump(mode="json")
