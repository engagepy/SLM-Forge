"""Agents, proposals (the approval gate), jobs and live event streams."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from slm.agents import actions
from slm.agents.base import set_proposal_status
from slm.api.common import SessionDep, get_or_404, project_or_404, topic_stream
from slm.data import scout_tools
from slm.data.format import guess_mapping
from slm.db import AgentEvent, Dataset, Job, Metric, Proposal
from slm.train.worker import log_tail, worker

router = APIRouter(prefix="/api", tags=["agents"])

# ── agents ──────────────────────────────────────────────────────────────────


class ScoutIn(BaseModel):
    request: str = ""


@router.post("/projects/{project_id}/agents/scout")
def run_scout(project_id: int, body: ScoutIn, s: Session = SessionDep) -> dict:
    project_or_404(s, project_id)
    return {"job_id": worker.submit("agent_scout", body.model_dump(), project_id).id}


class ObserveIn(BaseModel):
    use_llm: bool = True


@router.post("/projects/{project_id}/agents/observe")
def run_observer(project_id: int, body: ObserveIn, s: Session = SessionDep) -> dict:
    project_or_404(s, project_id)
    return {"job_id": worker.submit("agent_observer", body.model_dump(), project_id).id}


class PrepIn(BaseModel):
    dataset_id: int
    use_llm: bool = True


@router.post("/projects/{project_id}/agents/prep")
def run_prep(project_id: int, body: PrepIn, s: Session = SessionDep) -> dict:
    get_or_404(s, Dataset, body.dataset_id)
    return {"job_id": worker.submit("agent_prep", body.model_dump(), project_id).id}


class SynthIn(BaseModel):
    kind: str = "sft"
    count: int = 20
    focus: str = ""
    on_policy: bool = True


@router.post("/projects/{project_id}/agents/synthesize")
def run_synth(project_id: int, body: SynthIn, s: Session = SessionDep) -> dict:
    project_or_404(s, project_id)
    if body.kind not in ("sft", "preference") or not 1 <= body.count <= 200:
        raise HTTPException(422, "kind must be sft|preference and count 1-200")
    return {"job_id": worker.submit("synthesize", body.model_dump(), project_id).id}


@router.get("/projects/{project_id}/agents/events")
def agent_events(project_id: int, after: int = 0, limit: int = 200, s: Session = SessionDep) -> list[dict]:
    rows = s.exec(
        select(AgentEvent)
        .where(AgentEvent.project_id == project_id, AgentEvent.id > after)
        .order_by(AgentEvent.id.desc())
        .limit(min(limit, 1000))
    ).all()
    return [r.model_dump(mode="json") for r in reversed(rows)]


@router.get("/projects/{project_id}/stream")
async def project_stream(project_id: int):
    """Agent events and proposal changes for one project."""
    return EventSourceResponse(topic_stream(f"project:{project_id}"))


# ── HF dataset discovery (manual browsing, same tools the scout uses) ──────


@router.get("/datasets/search")
def hf_dataset_search(q: str, limit: int = 20) -> list[dict]:
    try:
        return scout_tools.search_datasets(q, limit=min(limit, 50))
    except Exception as e:
        raise HTTPException(502, f"Hugging Face search failed: {e}") from e


@router.get("/datasets/preview")
def hf_dataset_preview(repo_id: str, config: str | None = None, split: str | None = None) -> dict:
    try:
        data = scout_tools.preview_rows(repo_id, config, split, n=8)
    except Exception as e:
        raise HTTPException(502, f"Preview failed (gated or unsupported dataset?): {e}") from e

    return data | {"suggested_mapping": guess_mapping(data["columns"])}


class ImportIn(BaseModel):
    repo_id: str
    config: str | None = None
    split: str = "train"
    max_rows: int = 5000
    license: str = ""


@router.post("/projects/{project_id}/datasets/import")
def import_hf(project_id: int, body: ImportIn, s: Session = SessionDep) -> dict:
    project_or_404(s, project_id)
    return {"job_id": worker.submit("import_dataset", body.model_dump(), project_id).id}


# ── proposals ───────────────────────────────────────────────────────────────


@router.get("/projects/{project_id}/proposals")
def list_proposals(project_id: int, status: str | None = None, s: Session = SessionDep) -> list[dict]:
    q = select(Proposal).where(Proposal.project_id == project_id)
    if status:
        q = q.where(Proposal.status == status)
    return [p.model_dump(mode="json") for p in s.exec(q.order_by(Proposal.id.desc()).limit(200)).all()]


class ApproveIn(BaseModel):
    overrides: dict = {}


@router.post("/proposals/{proposal_id}/approve")
def approve_proposal(proposal_id: int, body: ApproveIn) -> dict:
    try:
        return actions.execute(proposal_id, body.overrides)
    except KeyError as e:
        raise HTTPException(404, "proposal not found") from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@router.post("/proposals/{proposal_id}/reject")
def reject_proposal(proposal_id: int, s: Session = SessionDep) -> dict:

    prop = get_or_404(s, Proposal, proposal_id)
    if prop.status != "pending":
        raise HTTPException(409, f"proposal is {prop.status}")
    return set_proposal_status(proposal_id, "rejected").model_dump(mode="json")


# ── jobs ────────────────────────────────────────────────────────────────────


@router.get("/jobs")
def list_jobs(
    project_id: int | None = None, kind: str | None = None, limit: int = 50, s: Session = SessionDep
) -> list[dict]:
    q = select(Job)
    if project_id is not None:
        q = q.where(Job.project_id == project_id)
    if kind:
        q = q.where(Job.kind.in_(kind.split(",")))
    return [j.model_dump(mode="json") for j in s.exec(q.order_by(Job.id.desc()).limit(min(limit, 500))).all()]


@router.get("/jobs/{job_id}")
def get_job(job_id: int, log_lines: int = 200, s: Session = SessionDep) -> dict:
    job = get_or_404(s, Job, job_id)
    metrics = s.exec(select(Metric).where(Metric.job_id == job_id).order_by(Metric.iteration)).all()
    return {
        "job": job.model_dump(mode="json"),
        "metrics": [m.model_dump(mode="json", exclude={"id", "job_id"}) for m in metrics],
        "log": log_tail(job, min(log_lines, 5000)),
    }


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int) -> dict:
    try:
        status = worker.cancel(job_id)
    except KeyError as e:
        raise HTTPException(404, f"job {job_id} not found") from e
    if status not in ("cancelled", "cancelling"):
        raise HTTPException(409, f"job {job_id} already {status}")
    return {"status": status}


@router.get("/jobs/{job_id}/stream")
async def job_stream(job_id: int):
    """Live logs, metrics and status for one job."""
    return EventSourceResponse(topic_stream(f"job:{job_id}"))


@router.get("/stream/jobs")
async def all_jobs_stream():
    return EventSourceResponse(topic_stream("jobs"))
