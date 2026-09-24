"""The Studio: chat with the Tuner (left) and the live pipeline canvas (right)."""

import copy

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, select
from sse_starlette.sse import EventSourceResponse

from slm import hardware
from slm.api.common import SessionDep, project_or_404, topic_stream
from slm.db import (
    Checkpoint,
    Dataset,
    DatasetVersion,
    Feedback,
    Job,
    ModelRecord,
    PreferencePair,
    StudioState,
    TunerMessage,
    now,
)
from slm.feedback import record_feedback
from slm.models import manage
from slm.tuner.session import tuner

router = APIRouter(prefix="/api/projects/{project_id}", tags=["studio"])


def _state(s: Session, pid: int) -> StudioState:
    st = s.get(StudioState, pid)
    if st is None:
        st = StudioState(project_id=pid)
        s.add(st)
        s.commit()
        s.refresh(st)
    return st


@router.get("/studio")
def studio_snapshot(project_id: int, s: Session = SessionDep) -> dict:
    """Everything the canvas renders, in one request."""
    p = project_or_404(s, project_id)
    st = _state(s, project_id)
    hw = hardware.detect()
    model = None
    if p.base_model:
        rec = s.exec(select(ModelRecord).where(ModelRecord.repo_id == p.base_model)).first()
        model = {"repo_id": p.base_model, "downloaded": rec is not None}
        if rec:
            shape = manage.read_shape(rec.local_path)
            est = hardware.estimate_inference(shape)
            model |= {
                "params": rec.params,
                "bits": rec.bits,
                "size_gb": rec.size_gb,
                "layers": shape.num_layers,
                "inference": est.to_dict(),
            }
    jobs = s.exec(select(Job).where(Job.project_id == project_id).order_by(Job.id.desc()).limit(30)).all()
    ckpts = s.exec(select(Checkpoint).where(Checkpoint.project_id == project_id).order_by(Checkpoint.id)).all()
    by_job = {j.id: j for j in jobs}
    return {
        "project": p.model_dump(mode="json"),
        "stage": st.stage,
        "note": st.note,
        "tuner_busy": tuner.is_busy(project_id),
        "autopilot": st.autopilot,
        "completed": st.completed,
        "hardware": hw.to_dict(),
        "model": model,
        "datasets": [
            d.model_dump(mode="json", exclude={"raw_path"})
            for d in s.exec(select(Dataset).where(Dataset.project_id == project_id)).all()
        ],
        "versions": [
            v.model_dump(mode="json", exclude={"path", "mapping"})
            for v in s.exec(
                select(DatasetVersion).where(DatasetVersion.project_id == project_id).order_by(DatasetVersion.id)
            ).all()
        ],
        "jobs": [j.model_dump(mode="json", exclude={"config", "log_path"}) for j in jobs],
        "checkpoints": [
            c.model_dump(mode="json")
            | {"warnings": (by_job.get(c.job_id).result or {}).get("warnings", []) if by_job.get(c.job_id) else []}
            for c in ckpts
        ],
        "samples": st.samples,
        "comparisons": st.comparisons[-20:],
        "feedback": {
            "judgements": len(s.exec(select(Feedback.id).where(Feedback.project_id == project_id)).all()),
            "pairs_ready": len(
                s.exec(
                    select(PreferencePair.id).where(
                        PreferencePair.project_id == project_id,
                        PreferencePair.approved == True,  # noqa: E712
                        PreferencePair.used_in_job_id == None,  # noqa: E711
                    )
                ).all()
            ),
        },
        "exports": [j.result for j in jobs if j.kind == "export" and j.status == "succeeded"],
    }


@router.get("/tuner/messages")
def tuner_messages(project_id: int, s: Session = SessionDep) -> list[dict]:
    rows = s.exec(select(TunerMessage).where(TunerMessage.project_id == project_id).order_by(TunerMessage.id)).all()
    return [m.model_dump(mode="json") for m in rows]


class MessageIn(BaseModel):
    text: str


@router.post("/tuner/message")
def tuner_message(project_id: int, body: MessageIn, s: Session = SessionDep) -> dict:
    project_or_404(s, project_id)
    if not body.text.strip():
        raise HTTPException(422, "empty message")
    tuner.send(project_id, body.text.strip())
    return {"queued": True}


@router.post("/tuner/start")
def tuner_start(project_id: int, s: Session = SessionDep) -> dict:
    """Kick off a new project's conversation (no-op if it has already started)."""
    p = project_or_404(s, project_id)
    if s.exec(select(TunerMessage.id).where(TunerMessage.project_id == project_id)).first():
        return {"started": False}
    goal = p.goal or "(not given yet)"
    tuner.send(
        project_id,
        f"[New project] The user just opened the Studio. Their description of the model: {goal}. "
        "Greet them in one line, restate the goal, and start building. Run the whole pipeline yourself "
        "without asking them anything.",
        role="event",
        meta={"kickoff": True},
    )
    return {"started": True}


@router.get("/tuner/stream")
async def tuner_stream(project_id: int):
    return EventSourceResponse(topic_stream(f"tuner:{project_id}"))


class JudgeIn(BaseModel):
    comparison_id: str
    choice: str  # a | b | tie | both_bad
    edited_answer: str = ""
    critique: str = ""


@router.post("/studio/judge")
def judge(project_id: int, body: JudgeIn, s: Session = SessionDep) -> dict:
    p = project_or_404(s, project_id)
    st = _state(s, project_id)
    # Deep copy: mutating the loaded dicts in place makes SQLAlchemy see "no change" and skip the write.
    comps = copy.deepcopy(st.comparisons)
    item = next((c for c in comps if c.get("id") == body.comparison_id), None)
    if item is None or item.get("status") != "pending":
        raise HTTPException(404, "comparison not found or already judged")
    try:
        out = record_feedback(
            project_id,
            prompt=item["prompt"],
            system=p.system_prompt,
            candidate_a=item["a"],
            candidate_b=item["b"],
            params_a=item.get("params_a"),
            params_b=item.get("params_b"),
            choice=body.choice,
            edited_answer=body.edited_answer,
            critique=body.critique,
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    item.update(status="judged", choice=body.choice, critique=body.critique, rewrote=bool(body.edited_answer.strip()))
    st.comparisons = comps
    flag_modified(st, "comparisons")
    st.updated_at = now()
    s.add(st)
    s.commit()
    batch_done = not any(c.get("status") == "pending" for c in comps)
    if batch_done:
        judged = [c for c in comps if c.get("status") == "judged"][-10:]
        lines = "; ".join(
            f"'{c['prompt'][:60]}': {c['choice']}"
            + (f" (critique: {c['critique'][:120]})" if c.get("critique") else "")
            + (" (rewrote answer)" if c.get("rewrote") else "")
            for c in judged
        )
        tuner.send(
            project_id,
            f"[Feedback] The user finished judging the comparisons. {lines}. Summarise what this says about the "
            "model and decide the next step.",
            role="event",
            meta={"feedback": True},
        )
    from slm.tuner.tools import canvas_changed

    canvas_changed(project_id)
    return out | {"batch_done": batch_done}


class AutopilotIn(BaseModel):
    on: bool


@router.post("/studio/autopilot")
def set_autopilot(project_id: int, body: AutopilotIn, s: Session = SessionDep) -> dict:
    """Switch autopilot. Turning it on wakes the Tuner to carry on from wherever things are."""
    project_or_404(s, project_id)
    st = _state(s, project_id)
    st.autopilot, st.stalled_nudges = body.on, 0
    if body.on:
        st.completed = False
    s.add(st)
    s.commit()
    if body.on and not tuner.is_busy(project_id):
        tuner.send(
            project_id, "[Autopilot on] Carry on from where things stand.", role="event", meta={"autopilot": True}
        )
    return {"autopilot": st.autopilot}


# ── sessions across projects ────────────────────────────────────────────────

sessions_router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@sessions_router.get("")
def list_sessions() -> dict:
    """Every project's live state, plus what the GPU is doing and what's queued."""
    from slm.sessions import overview

    return overview()


class StopIn(BaseModel):
    cancel_jobs: bool = True


@sessions_router.post("/{project_id}/stop")
def stop_session(project_id: int, body: StopIn, s: Session = SessionDep) -> dict:
    from slm.sessions import stop_project

    project_or_404(s, project_id)
    return stop_project(project_id, body.cancel_jobs)


@sessions_router.post("/{project_id}/resume")
def resume_session(project_id: int, s: Session = SessionDep) -> dict:
    from slm.sessions import resume_project

    project_or_404(s, project_id)
    return resume_project(project_id)
