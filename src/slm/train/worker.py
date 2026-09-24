"""Background job execution with three lanes, each running one job at a time.

- gpu:   training, fusing, export. Evicts the inference model first.
- io:    downloads, dataset imports and preparation.
- agent: LLM agent runs (scout, prep, observer, synthesis), which can take minutes.
"""

import queue
import threading
import traceback
from collections.abc import Callable
from pathlib import Path

from sqlmodel import Session, select

from slm.config import get_settings
from slm.db import Job, Metric, engine, now
from slm.events import bus
from slm.inference.engine import engine as inference_engine
from slm.train.runner import Cancelled, ParsedMetric, tail

GPU_KINDS = {"sft", "dpo", "export"}
AGENT_KINDS = {"agent_scout", "agent_prep", "agent_observer", "synthesize"}


class JobContext:
    def __init__(self, job: Job, worker: "JobWorker") -> None:
        self.job_id = job.id
        self.project_id = job.project_id
        self.kind = job.kind
        self.config = dict(job.config)
        self.result: dict = {}
        self.run_dir = get_settings().runs_dir / f"job-{job.id:05d}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "log.txt"
        self._worker = worker

    def log(self, line: str) -> None:
        bus.publish(f"job:{self.job_id}", {"type": "log", "line": line})

    def note(self, line: str) -> None:
        """A log line written by us (not the subprocess): goes to the log file too."""
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        self.log(line)

    def metric(self, m: ParsedMetric) -> None:
        with Session(engine()) as s:
            s.add(Metric(job_id=self.job_id, iteration=m.iteration, split=m.split, values=m.values))
            s.commit()
        bus.publish(
            f"job:{self.job_id}",
            {"type": "metric", "iteration": m.iteration, "split": m.split, "values": m.values},
        )

    def progress(self, current: int, total: int) -> None:
        self.result["progress"] = {"current": current, "total": total}
        bus.publish(f"job:{self.job_id}", {"type": "progress", "current": current, "total": total})

    def should_cancel(self) -> bool:
        return self.job_id in self._worker._cancel_requested


Handler = Callable[[JobContext], None]


class JobWorker:
    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}
        self._lanes = {"gpu": queue.Queue(), "io": queue.Queue(), "agent": queue.Queue()}
        self._cancel_requested: set[int] = set()
        self._running: dict[str, int | None] = {"gpu": None, "io": None, "agent": None}
        self._started = False
        self._finish_hooks: list[Callable[[Job], None]] = []

    def on_finish(self, fn: Callable[[Job], None]) -> None:
        """Call `fn(job)` after any job reaches a terminal status (e.g. to wake the Tuner)."""
        self._finish_hooks.append(fn)

    def register(self, kind: str) -> Callable[[Handler], Handler]:
        def deco(fn: Handler) -> Handler:
            self._handlers[kind] = fn
            return fn

        return deco

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        with Session(engine()) as s:
            # A restart orphans running jobs; queued ones can simply resume.
            for job in s.exec(select(Job).where(Job.status == "running")).all():
                job.status, job.error, job.finished_at = "failed", "Server restarted mid-job", now()
                s.add(job)
            s.commit()
            queued = s.exec(select(Job).where(Job.status == "queued").order_by(Job.id)).all()
            for job in queued:
                self._lanes[self._lane(job.kind)].put(job.id)
        for lane in self._lanes:
            threading.Thread(target=self._loop, args=(lane,), daemon=True, name=f"jobs-{lane}").start()

    @staticmethod
    def _lane(kind: str) -> str:
        if kind in GPU_KINDS:
            return "gpu"
        return "agent" if kind in AGENT_KINDS else "io"

    def submit(self, kind: str, config: dict, project_id: int | None = None) -> Job:
        if kind not in self._handlers:
            raise ValueError(f"unknown job kind: {kind}")
        with Session(engine()) as s:
            job = Job(kind=kind, config=config, project_id=project_id)
            s.add(job)
            s.commit()
            s.refresh(job)
            job.log_path = str(get_settings().runs_dir / f"job-{job.id:05d}" / "log.txt")
            s.add(job)
            s.commit()
            s.refresh(job)
        self._publish_status(job)
        self._lanes[self._lane(kind)].put(job.id)
        return job

    def cancel(self, job_id: int) -> str:
        with Session(engine()) as s:
            job = s.get(Job, job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status == "queued":
                job.status, job.finished_at = "cancelled", now()
                s.add(job)
                s.commit()
                s.refresh(job)
                self._publish_status(job)
                return "cancelled"
            if job.status == "running":
                self._cancel_requested.add(job_id)
                return "cancelling"
            return job.status

    def status(self) -> dict:
        return {
            "running": dict(self._running),
            "queued": {k: q.qsize() for k, q in self._lanes.items()},
            "inference_blocked": inference_engine.blocked,
        }

    def _publish_status(self, job: Job) -> None:
        ev = {
            "type": "status",
            **job.model_dump(mode="json", include={"id", "kind", "status", "project_id", "error", "result"}),
        }
        bus.publish(f"job:{job.id}", ev)
        bus.publish("jobs", ev)

    def _loop(self, lane: str) -> None:
        q = self._lanes[lane]
        while True:
            job_id = q.get()
            try:
                self._running[lane] = job_id
                self._run(job_id, lane)
            except Exception:  # never let the lane thread die
                traceback.print_exc()
            finally:
                self._running[lane] = None
                self._cancel_requested.discard(job_id)

    def _run(self, job_id: int, lane: str) -> None:
        with Session(engine()) as s:
            job = s.get(Job, job_id)
            if job is None or job.status != "queued":
                return
            job.status, job.started_at = "running", now()
            s.add(job)
            s.commit()
            s.refresh(job)
            self._publish_status(job)
            ctx = JobContext(job, self)

        if lane == "gpu":
            inference_engine.block(f"job {job_id} ({ctx.kind}) is using the GPU")
        status, error = "succeeded", ""
        try:
            self._handlers[ctx.kind](ctx)
        except Cancelled:
            status = "cancelled"
            ctx.note("Cancelled by user.")
        except Exception as e:
            status, error = "failed", f"{e.__class__.__name__}: {e}"
            ctx.note(traceback.format_exc())
        finally:
            if lane == "gpu":
                inference_engine.unblock()

        with Session(engine()) as s:
            job = s.get(Job, job_id)
            job.status, job.error, job.finished_at = status, error, now()
            job.result = ctx.result
            s.add(job)
            s.commit()
            s.refresh(job)
            self._publish_status(job)
        for hook in self._finish_hooks:
            try:
                hook(job)
            except Exception:  # a hook must never break the worker
                traceback.print_exc()


worker = JobWorker()


def log_tail(job: Job, n: int = 200) -> str:

    return tail(Path(job.log_path), n) if job.log_path else ""
