"""Playground generation, A/B comparison, human feedback and example review."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from slm.api.common import SessionDep, get_or_404, project_or_404, sse
from slm.db import Checkpoint, Feedback, PreferencePair, Project, SftExample
from slm.feedback import record_feedback
from slm.inference.engine import EngineBusy, SamplingParams
from slm.inference.engine import engine as infer
from slm.models import manage

router = APIRouter(prefix="/api/projects/{project_id}", tags=["feedback"])


class GenerateIn(BaseModel):
    messages: list[dict]
    params: SamplingParams = SamplingParams()
    target: str = "current"  # current | base | checkpoint:<id>


def _target(s: Session, p: Project, target: str) -> dict:
    if target == "base":
        path = manage.local_path_for(p.base_model or "")
        if not path:
            raise HTTPException(409, "Base model not downloaded")
        return {"model_path": path, "adapter_path": None}
    if target.startswith("checkpoint:"):
        c = get_or_404(s, Checkpoint, int(target.split(":", 1)[1]))
        if c.fused_path:
            return {"model_path": c.fused_path, "adapter_path": None}
        return {"model_path": c.base_model_path, "adapter_path": c.adapter_path}
    path = p.current_model_path or manage.local_path_for(p.base_model or "")
    if not path:
        raise HTTPException(409, "Download the project's base model first")
    return {"model_path": path, "adapter_path": p.current_adapter_path}


def _with_system(p: Project, messages: list[dict]) -> list[dict]:
    if p.system_prompt and not any(m["role"] == "system" for m in messages):
        return [{"role": "system", "content": p.system_prompt}, *messages]
    return messages


def _stream_events(gens: list[tuple[str | None, list[dict], SamplingParams, dict]]):
    """Run one or more generations back to back, tagging events with a candidate label."""
    for label, messages, params, where in gens:
        try:
            for ev in infer.stream(messages, params, **where):
                yield sse(ev | ({"candidate": label} if label else {}))
        except EngineBusy as e:
            yield sse({"type": "error", "message": f"GPU busy: {e}"})
            return
        except Exception as e:
            yield sse({"type": "error", "message": f"{e.__class__.__name__}: {e}"})
            return
    yield sse({"type": "end"})


@router.post("/generate")
def generate(project_id: int, body: GenerateIn, s: Session = SessionDep):
    p = project_or_404(s, project_id)
    where = _target(s, p, body.target)
    messages = _with_system(p, body.messages)
    return EventSourceResponse(iterate_in_threadpool(_stream_events([(None, messages, body.params, where)])))


class CompareIn(BaseModel):
    prompt: str
    system: str | None = None
    params_a: SamplingParams = SamplingParams(temperature=0.7, seed=1)
    params_b: SamplingParams = SamplingParams(temperature=1.0, seed=2)
    target_a: str = "current"
    target_b: str = "current"


@router.post("/compare")
def compare(project_id: int, body: CompareIn, s: Session = SessionDep):
    """Two candidates for the feedback screen, streamed as candidate a then b."""
    p = project_or_404(s, project_id)
    system = body.system if body.system is not None else p.system_prompt
    messages = [{"role": "user", "content": body.prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    gens = [
        ("a", messages, body.params_a, _target(s, p, body.target_a)),
        ("b", messages, body.params_b, _target(s, p, body.target_b)),
    ]
    return EventSourceResponse(iterate_in_threadpool(_stream_events(gens)))


class FeedbackIn(BaseModel):
    prompt: str
    system: str = ""
    candidate_a: str
    candidate_b: str
    params_a: dict = {}
    params_b: dict = {}
    choice: str  # a | b | tie | both_bad
    edited_answer: str = ""
    critique: str = ""
    model_ref: str = ""


@router.post("/feedback")
def submit_feedback(project_id: int, body: FeedbackIn, s: Session = SessionDep) -> dict:
    project_or_404(s, project_id)
    try:
        return record_feedback(project_id, **body.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.get("/feedback")
def list_feedback(project_id: int, limit: int = 50, s: Session = SessionDep) -> list[dict]:
    rows = s.exec(
        select(Feedback).where(Feedback.project_id == project_id).order_by(Feedback.id.desc()).limit(min(limit, 500))
    ).all()
    return [r.model_dump(mode="json") for r in rows]


@router.get("/examples")
def list_examples(project_id: int, status: str = "pending", s: Session = SessionDep) -> dict:
    """status: pending (awaiting review) | ready (approved, untrained) | all"""
    q_pairs = select(PreferencePair).where(PreferencePair.project_id == project_id)
    q_sft = select(SftExample).where(SftExample.project_id == project_id)
    if status == "pending":
        q_pairs = q_pairs.where(PreferencePair.approved == False)  # noqa: E712
        q_sft = q_sft.where(SftExample.approved == False)  # noqa: E712
    elif status == "ready":
        q_pairs = q_pairs.where(PreferencePair.approved == True, PreferencePair.used_in_job_id == None)  # noqa: E711,E712
        q_sft = q_sft.where(SftExample.approved == True, SftExample.used_in_job_id == None)  # noqa: E711,E712
    return {
        "pairs": [
            r.model_dump(mode="json") for r in s.exec(q_pairs.order_by(PreferencePair.id.desc()).limit(500)).all()
        ],
        "sft": [r.model_dump(mode="json") for r in s.exec(q_sft.order_by(SftExample.id.desc()).limit(500)).all()],
    }


class ReviewIn(BaseModel):
    pair_ids: list[int] = []
    sft_ids: list[int] = []
    approve: bool = True  # False deletes them


@router.post("/examples/review")
def review_examples(project_id: int, body: ReviewIn, s: Session = SessionDep) -> dict:
    n = 0
    for model, ids in ((PreferencePair, body.pair_ids), (SftExample, body.sft_ids)):
        for i in ids:
            row = s.get(model, i)
            if row is None or row.project_id != project_id or row.used_in_job_id is not None:
                continue
            if body.approve:
                row.approved = True
                s.add(row)
            else:
                s.delete(row)
            n += 1
    s.commit()
    return {"updated": n}


class PairEdit(BaseModel):
    prompt: str | None = None
    chosen: str | None = None
    rejected: str | None = None


@router.patch("/examples/pairs/{pair_id}")
def edit_pair(project_id: int, pair_id: int, body: PairEdit, s: Session = SessionDep) -> dict:
    row = get_or_404(s, PreferencePair, pair_id)
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(row, k, v)
    s.add(row)
    s.commit()
    s.refresh(row)
    return row.model_dump(mode="json")


class SftEdit(BaseModel):
    messages: list[dict]


@router.patch("/examples/sft/{example_id}")
def edit_sft(project_id: int, example_id: int, body: SftEdit, s: Session = SessionDep) -> dict:
    row = get_or_404(s, SftExample, example_id)
    row.messages = body.messages
    s.add(row)
    s.commit()
    s.refresh(row)
    return row.model_dump(mode="json")
