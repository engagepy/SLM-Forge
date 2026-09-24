"""A machine-wide view of every project: what's running, what's queued, and what each is doing.

The GPU lane trains one model at a time: training runs share this Mac's unified memory, so two
at once would compete for it and likely fail. Everything else queues. Downloads, data prep and
agent turns run alongside.
"""

from sqlmodel import Session, select

from slm import hardware
from slm.db import Job, Metric, Project, StudioState, engine
from slm.train.worker import GPU_KINDS

JOB_LABEL = {
    "sft": "training",
    "dpo": "preference training",
    "fuse": "merging adapters",
    "export": "exporting",
    "download": "downloading a model",
    "import_dataset": "importing data",
    "prepare_dataset": "preparing data",
    "synthesize": "writing examples",
    "agent_scout": "searching for data",
    "agent_prep": "planning data prep",
    "agent_observer": "reviewing feedback",
}


def _progress(s: Session, job: Job) -> dict:
    """Percent done and minutes left, from the latest reported training speed."""
    prog = (job.result or {}).get("progress") or {}
    total = prog.get("total") or (job.result or {}).get("total_iters") or 0
    current = prog.get("current") or 0
    last = s.exec(
        select(Metric).where(Metric.job_id == job.id, Metric.split == "train").order_by(Metric.iteration.desc())
    ).first()
    if last:
        current = max(current, last.iteration)
    out = {"current": current, "total": total, "percent": round(100 * current / total) if total else None}
    speed = last.values.get("it_per_sec") if last else None
    if speed and total > current:
        out["minutes_left"] = max(1, round((total - current) / speed / 60))
    return out


def overview() -> dict:
    from slm.tuner.session import tuner

    hw = hardware.detect()
    with Session(engine()) as s:
        projects = s.exec(select(Project).order_by(Project.id.desc())).all()
        states = {st.project_id: st for st in s.exec(select(StudioState)).all()}
        active = s.exec(select(Job).where(Job.status.in_(["running", "queued"])).order_by(Job.id)).all()
        gpu_running = next((j for j in active if j.kind in GPU_KINDS and j.status == "running"), None)
        gpu_queue = [j for j in active if j.kind in GPU_KINDS and j.status == "queued"]
        names = {p.id: p.name for p in projects}

        def brief(j: Job) -> dict:
            out = {"job_id": j.id, "project_id": j.project_id, "project": names.get(j.project_id), "kind": j.kind,
                   "label": JOB_LABEL.get(j.kind, j.kind), "status": j.status}  # fmt: skip
            if j.status == "running" and j.kind in ("sft", "dpo"):
                out["progress"] = _progress(s, j)
            return out

        sessions = []
        for p in projects:
            st = states.get(p.id)
            mine = [j for j in active if j.project_id == p.id]
            running = next((j for j in mine if j.status == "running"), None)
            queued = next((j for j in mine if j.status == "queued"), None)
            if running:
                state = "running"
            elif queued:
                state = "queued"
            elif tuner.is_busy(p.id):
                state = "thinking"
            elif st and st.completed:
                state = "done"
            elif st and not st.autopilot:
                state = "paused"
            else:
                state = "idle"
            item = {
                "project_id": p.id,
                "name": p.name,
                "goal": p.goal,
                "stage": st.stage if st else "goal",
                "autopilot": st.autopilot if st else True,
                "completed": st.completed if st else False,
                "state": state,
                "created_at": p.created_at.isoformat(),
            }
            if running:
                item["job"] = brief(running)
            elif queued:
                item["job"] = brief(queued)
                if queued.kind in GPU_KINDS:
                    item["job"]["queue_position"] = [j.id for j in gpu_queue].index(queued.id) + 1
            sessions.append(item)

        return {
            "capacity": {
                "chip": hw.chip,
                "memory_gb": hw.total_memory_gb,
                "ml_budget_gb": round(hw.budget_gb, 1),
                "gpu_slots": 1,
                "explanation": (
                    f"This Mac trains one model at a time: training uses most of its "
                    f"{hw.total_memory_gb:.0f} GB unified memory, so two runs at once would compete for it and "
                    "likely fail. Other training runs wait in a queue. Downloads, data preparation and "
                    "the agents' thinking run in parallel."
                ),
                "gpu_running": brief(gpu_running) if gpu_running else None,
                "gpu_queue": [brief(j) for j in gpu_queue],
            },
            "sessions": sessions,
        }


def stop_project(project_id: int, cancel_jobs: bool = True) -> dict:
    """Pause a project's autopilot and, optionally, cancel its queued and running jobs."""
    from slm.train.worker import worker

    cancelled = []
    with Session(engine()) as s:
        st = s.get(StudioState, project_id) or StudioState(project_id=project_id)
        st.autopilot = False
        st.stalled_nudges = 0
        s.add(st)
        s.commit()
        jobs = s.exec(select(Job).where(Job.project_id == project_id, Job.status.in_(["running", "queued"]))).all()
        ids = [j.id for j in jobs]
    if cancel_jobs:
        for jid in ids:
            worker.cancel(jid)
            cancelled.append(jid)
    return {"project_id": project_id, "autopilot": False, "cancelled_jobs": cancelled}


def resume_project(project_id: int) -> dict:
    from slm.tuner.session import tuner

    with Session(engine()) as s:
        st = s.get(StudioState, project_id) or StudioState(project_id=project_id)
        st.autopilot, st.completed, st.stalled_nudges = True, False, 0
        s.add(st)
        s.commit()
    if not tuner.is_busy(project_id):
        tuner.send(
            project_id, "[Autopilot on] Carry on from where things stand.", role="event", meta={"autopilot": True}
        )
    return {"project_id": project_id, "autopilot": True}
