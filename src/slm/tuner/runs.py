"""Starting a job for the Tuner and waiting for a quick one, shared by the tools and confirm.py."""

import time

from sqlmodel import Session

from slm.db import Job, engine
from slm.events import canvas_changed, shutting_down
from slm.train.worker import worker
from slm.tuner import confirm

# Jobs the Tuner starts and does not wait for: it gets woken when they finish.
NOTIFY_KINDS = {"download", "sft", "dpo", "export", "gguf", "hf_upload"}


def submit_job(kind: str, config: dict, project_id: int) -> Job:
    from slm.tuner.session import tuner

    if tuner.is_halted(project_id):
        raise ValueError(confirm.STOPPED)
    job = worker.submit(kind, config | {"origin": "tuner", "notify": kind in NOTIFY_KINDS}, project_id)
    canvas_changed(project_id)
    return job


def wait_job(job_id: int, timeout: float) -> Job:
    """Wait for a quick job; the UI shows its progress meanwhile."""
    deadline = time.time() + timeout
    while True:
        with Session(engine()) as s:
            job = s.get(Job, job_id)
            if job.status in ("succeeded", "failed", "cancelled") or time.time() > deadline:
                return job
        if shutting_down.wait(1):
            return job  # the server is stopping: don't keep a thread alive for this
