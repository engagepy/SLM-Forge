"""Playground generation, A/B comparison, human feedback and example review."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from slm.api.common import SessionDep, get_or_404, project_or_404, sse
from slm.db import Checkpoint, Job, PreferencePair, Project, SftExample, awaiting_review, ready
from slm.feedback import record_feedback
from slm.inference.engine import EngineBusy, SamplingParams
from slm.inference.engine import engine as infer
from slm.models import manage

router = APIRouter(prefix="/api/projects/{project_id}", tags=["feedback"])


class GenerateIn(BaseModel):
    messages: list[dict]
    params: SamplingParams = SamplingParams()
    target: str = "current"  # current | base | checkpoint:<id> | export:<job id>


def _target(s: Session, p: Project, target: str) -> dict:
    if target == "base":
        path = manage.local_path_for(p.base_model or "")
        if not path:
            raise HTTPException(409, "Base model not downloaded")
        return {"model_path": path, "adapter_path": None}
    if target.startswith("export:"):
        # The finished, exported model exactly as it sits on disk (fused and quantized).
        job = s.get(Job, int(target.split(":", 1)[1]))
        if job is None or job.project_id != p.id or job.kind != "export" or job.status != "succeeded":
            raise HTTPException(404, "No such export in this project")
        path = (job.result or {}).get("path")
        if not path or not Path(path).is_dir():
            raise HTTPException(409, f"The exported model is no longer at {path}")
        return {"model_path": path, "adapter_path": None}
    if target.startswith("checkpoint:"):
        c = get_or_404(s, Checkpoint, int(target.split(":", 1)[1]))
        if c.project_id != p.id:
            raise HTTPException(404, "No such checkpoint in this project")
        if c.fused_path:
            return {"model_path": c.fused_path, "adapter_path": None}
        return {"model_path": c.base_model_path, "adapter_path": c.adapter_path}
    path = manage.serving_path(p)
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
    # An export is tried exactly as it runs elsewhere: its own template supplies the system prompt
    # (or doesn't), so Try it can't make an unbaked export look better than the CLI.
    messages = body.messages if body.target.startswith("export:") else _with_system(p, body.messages)
    return EventSourceResponse(iterate_in_threadpool(_stream_events([(None, messages, body.params, where)])))


SUGGEST_SYSTEM = """You write test inputs for a small language model that was fine-tuned for one specific
job. Someone is trying the finished model and wants inputs that show whether it really does the
job. Write exactly four, each under 160 characters, in the form a real user would type:
- three ON-GOAL inputs of varied difficulty (easy, typical, tricky), the kind of thing the model is
  for, phrased differently from the ones already used;
- one input where the CORRECT behaviour is to produce the empty/negative result the system prompt
  defines (or to say it can't help): off-topic text, a greeting, or a near-miss that mentions the
  domain without meeting the condition. Set kind="should-not".
Never repeat or lightly rephrase an input from the list already used. For each, say in a few words
what a correct answer looks like (expect)."""

SUGGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": ["on-goal", "should-not"]},
                    "expect": {"type": "string"},
                },
                "required": ["text", "kind", "expect"],
            },
        }
    },
    "required": ["prompts"],
}


class SuggestIn(BaseModel):
    used: list[str] = []


@router.post("/try/prompts")
def suggest_prompts(project_id: int, body: SuggestIn, s: Session = SessionDep) -> dict:
    """Four fresh inputs for trying the model: three on-goal, one that should yield the empty or
    negative result. Written by the teacher model from the goal and system prompt (one API call)."""
    from slm.agents.provider import get_provider

    p = project_or_404(s, project_id)
    used = [u.strip() for u in body.used if u.strip()][-40:]
    user = (
        f"Goal: {p.goal}\nSystem prompt the model runs with: {p.system_prompt or '(none)'}\n\n"
        f"Already used (don't repeat):\n" + ("\n".join(f"- {u}" for u in used) or "- (none)")
    )
    try:
        out = get_provider().json(SUGGEST_SYSTEM, user, SUGGEST_SCHEMA)
    except Exception as e:
        raise HTTPException(502, f"Couldn't write suggestions: {e}") from e
    seen = {u.lower() for u in used}
    prompts = []
    for it in out.get("prompts", []):
        text = str(it.get("text", "")).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            kind = "should-not" if it.get("kind") == "should-not" else "on-goal"
            prompts.append({"text": text[:200], "kind": kind, "expect": str(it.get("expect", ""))[:120]})
    return {"prompts": prompts[:4]}


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


@router.get("/examples")
def list_examples(project_id: int, status: str = "pending", s: Session = SessionDep) -> dict:
    """status: pending (awaiting review) | ready (approved, untrained) | all"""
    q_pairs = select(PreferencePair).where(PreferencePair.project_id == project_id)
    q_sft = select(SftExample).where(SftExample.project_id == project_id)
    if status in ("pending", "ready"):
        filters = awaiting_review if status == "pending" else ready
        q_pairs, q_sft = q_pairs.where(*filters(PreferencePair)), q_sft.where(*filters(SftExample))
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
