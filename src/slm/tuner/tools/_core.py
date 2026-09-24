"""The Tuner's tools: what they share. The context, the `tool` decorator (and its registry), the helpers that
open a proposal, wait on a job or mutate the studio, and the briefs every module returns."""

import asyncio
import functools
from contextlib import contextmanager
from dataclasses import dataclass

from agents import RunContextWrapper, function_tool
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session

from slm.db import (
    Dataset,
    DatasetVersion,
    Job,
    Project,
    engine,
    now,
    studio_state,
)
from slm.events import canvas_changed, shutting_down
from slm.tuner import confirm
from slm.tuner.runs import submit_job, wait_job

# The tools' names for starting a job and waiting on a quick one (runs.py, shared with confirm).
_submit, _wait = submit_job, wait_job

STAGES = ("goal", "data", "model", "train", "evaluate", "refine", "export")


@dataclass
class TunerContext:
    project_id: int


Ctx = RunContextWrapper[TunerContext]


REGISTRY: list = []  # every @tool in definition order; ALL_TOOLS in tools/__init__ must list them all


def tool(fn):
    """Register a Tuner tool. Tools block (HTTP, waiting on jobs, generating on the GPU), so each
    runs in a worker thread to keep the shared event loop, and the agent's streaming, responsive.
    functools.wraps keeps the signature and docstring the SDK builds the tool schema from."""

    @functools.wraps(fn)
    async def run_in_thread(*args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    t = function_tool(run_in_thread, strict_mode=False)
    REGISTRY.append(t)
    return t


def _own_dataset(s: Session, pid: int, dataset_id: int) -> Dataset:
    ds = s.get(Dataset, dataset_id)
    if ds is None or ds.project_id != pid:
        raise ValueError("no such dataset in this project")
    return ds


@contextmanager
def studio(pid: int, *json_fields: str):
    """Mutate the project's StudioState and show it: commit on exit, then refresh the canvas.
    Name the JSON columns you assign to, so SQLAlchemy writes them (it can't see in-place edits)."""
    with Session(engine()) as s:
        st = studio_state(s, pid)
        yield st
        for f in json_fields:
            flag_modified(st, f)
        st.updated_at = now()
        s.add(st)
        s.commit()
    canvas_changed(pid)


def _version_brief(v: DatasetVersion) -> dict:
    """What the Tuner needs to know about a prepared dataset version."""
    tokens, cleaning = v.token_stats or {}, v.cleaning_report or {}
    return {
        "version_id": v.id,
        "kind": v.kind,
        "train": v.n_train,
        "valid": v.n_valid,
        "test": v.n_test,
        "cleaning": cleaning.get("dropped"),
        "tokens": {k: tokens.get(k) for k in ("p50", "p95", "max", "over_max_seq_length", "estimated")},
        "length": cleaning.get("length"),
        "sampled": cleaning.get("sampled"),
    }


# Tools that spend (API tokens, downloads, GPU time). Inside a round the user set in motion they run
# freely, because reaching the goal is the mandate. Outside one (the project is finished, or autopilot
# is paused) each call is a proposal the user confirms; confirming grants that one call.
MAX_UNREVIEWED = 150  # synthetic examples nobody has looked at yet: review before writing more


def _spend(pid: int, tool_name: str, kind: str, title: str, reason: str, details: dict, args: dict) -> dict | None:
    """None: go ahead. A dict: the proposal to return instead (the user decides)."""
    from slm.tuner.session import tuner

    if tuner.is_halted(pid):
        raise ValueError(confirm.STOPPED)
    if confirm.take_grant(pid, tool_name, args):
        return None
    if confirm.pending(pid):
        raise ValueError(
            "A proposal is already waiting for the user's decision. Don't start other costly work meanwhile: "
            "answer them, and wait for their go-ahead."
        )
    if not confirm.needs_go_ahead(pid):
        return None
    if not reason.strip():
        raise ValueError(
            f"{tool_name} costs something and the user hasn't set a round in motion, so it needs their "
            "go-ahead: call it again with a plain-words `reason` for the card (what it's for, what it costs)."
        )
    return confirm.propose(pid, kind, title, reason, details, {"tool": tool_name, "args": args})


def _check_stop(pid: int) -> None:
    """Between costly calls in a loop: stop when the user halted or the server is going down."""
    from slm.tuner.session import tuner

    if shutting_down.is_set() or tuner.is_halted(pid):
        raise ValueError(confirm.STOPPED)


def _job_brief(j: Job) -> dict:
    out = {"job_id": j.id, "kind": j.kind, "status": j.status}
    if j.error:
        out["error"] = j.error[:300]
    r = j.result or {}
    for k in (
        "progress",
        "metrics",
        "warnings",
        "dataset_version_id",
        "dataset_id",
        "rows",
        "path",
        "size_gb",
        "min_ram_gb",
    ):
        if k in r:
            out[k] = r[k]
    return out


def _served_checkpoint_id(s: Session, project: Project) -> int | None:
    """The checkpoint the project serves right now, or None for the plain base model."""
    from slm.train.jobs import served_checkpoint

    c = served_checkpoint(s, project)
    return c.id if c else None
