"""Runs start only with the user's go-ahead.

The Tuner can't start a download, a training run or an export by itself. The tool records a
proposal (`StudioState.pending_action`) and the Studio shows it as a card with "Go ahead" and
"Not now". Confirming (the button, or a plain "yes" typed in the chat) submits the job and tells
the Tuner; declining tells it too, so it can ask what the user would rather do. One proposal is
pending at a time; a new one replaces the old.
"""

import json
import re
import time

from sqlmodel import Session, select

from slm.db import Checkpoint, Project, engine, now, studio_state
from slm.events import canvas_changed
from slm.models import manage
from slm.sessions import overview

# Every kind of card: (the tool a confirmed spend grants one call of, or None for a run that
# submits a job when confirmed; where the canvas goes once it starts). See tools._spend.
KINDS = {
    "model": (None, None),
    "sft": (None, "train"),
    "dpo": (None, "refine"),
    "export": (None, "export"),
    "synthesize": ("generate_synthetic_examples", "data"),
    "review": ("ai_review_answers", "refine"),
    "import": ("import_dataset", "data"),
    "evaluate": ("evaluate_model", "evaluate"),
    "scout": ("scout_datasets", "data"),
    "prep": ("plan_preparation", "data"),
}
ACTION_KINDS = tuple(KINDS)
SPEND_TOOL = {kind: tool for kind, (tool, _) in KINDS.items() if tool}
STAGE = {kind: stage for kind, (_, stage) in KINDS.items() if stage}
STOPPED = "The user stopped this project. Don't start new work unless they ask you to."
SWITCH_BASE = "Training has already started on another base model; start a new project to switch."

_YES = re.compile(
    r"^\s*(y|yes|yep|yeah|yup|ok|okay|sure|go|go ahead|go for it|do it|start|start it|confirm|confirmed|"
    r"sounds good|looks good|let'?s go|proceed|approved?)\s*[.!👍]*\s*$",
    re.IGNORECASE,
)


def is_plain_yes(text: str) -> bool:
    """A chat reply that only says yes: it confirms the pending proposal like the button does."""
    return bool(_YES.match(text))


def base_model_locked(s: Session, project: Project, repo_id: str) -> bool:
    """A project's base model can't change once something has been trained on it."""
    trained = s.exec(select(Checkpoint.id).where(Checkpoint.project_id == project.id)).first()
    return project.base_model != repo_id and trained is not None


def _same_args(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def take_grant(pid: int, tool_name: str, args: dict) -> bool:
    """Use up the go-ahead the user gave for exactly this call: same tool, same arguments. A call
    with different arguments (200 examples where 20 were approved) gets no grant and proposes again."""
    with Session(engine()) as s:
        st = studio_state(s, pid)
        for g in st.granted:
            if isinstance(g, dict) and g.get("tool") == tool_name and _same_args(g.get("args", {}), args):
                st.granted = [x for x in st.granted if x is not g]
                s.add(st)
                s.commit()
                return True
        return False


def clear_grants(pid: int) -> None:
    with Session(engine()) as s:
        st = studio_state(s, pid)
        if st.granted:
            st.granted = []
            s.add(st)
            s.commit()


def needs_go_ahead(pid: int) -> bool:
    """Is the Tuner outside a round the user set in motion? Then spending needs a fresh go-ahead.
    A round runs from the kickoff (or a confirmed action) to the export; while it's on, the goal
    itself is the mandate and the Tuner may spend to reach it."""
    with Session(engine()) as s:
        st = studio_state(s, pid)
        return st.completed or not st.autopilot


def pending(pid: int) -> dict:
    with Session(engine()) as s:
        return dict(studio_state(s, pid).pending_action or {})


def _store(pid: int, action: dict) -> None:
    with Session(engine()) as s:
        st = studio_state(s, pid)
        st.pending_action, st.updated_at = action, now()  # a new dict, so SQLAlchemy sees the change
        s.add(st)
        s.commit()


def propose(pid: int, kind: str, title: str, reason: str, details: dict, payload: dict) -> dict:
    """Record a run for the user to confirm. Returns what the Tuner should do meanwhile."""
    from slm.tuner.session import tuner

    if kind not in ACTION_KINDS:
        raise ValueError(f"unknown action {kind!r}")
    if tuner.is_halted(pid):
        raise ValueError(STOPPED)
    action = {
        "id": f"a{int(time.time() * 1000)}",
        "kind": kind,
        "title": title[:120],
        "reason": reason.strip()[:600],
        "details": details,
        "payload": payload,
        "created_at": now().isoformat(),
    }
    _store(pid, action)
    canvas_changed(pid)
    return {
        "status": "waiting for the user's confirmation",
        "proposal": action["title"],
        "note": (
            "Nothing has started. The user now sees a card with 'Go ahead' and 'Not now'. End your turn "
            "with a short message saying what you propose and why, and ask them to confirm. You'll be told "
            "when they decide. Don't propose it again or try to start it another way meanwhile."
        ),
    }


def _take(pid: int, action_id: str | None) -> dict:
    """Remove the pending proposal for a decision the user just made, and let the Tuner act again."""
    from slm.tuner.session import reset_stall, tuner

    action = pending(pid)
    if not action or (action_id and action.get("id") != action_id):
        raise LookupError("Nothing is waiting for confirmation (it may have been replaced).")
    _store(pid, {})
    clear_grants(pid)  # a decision supersedes any go-ahead still lying around
    tuner.unhalt(pid)  # deciding is the user taking part again
    reset_stall(pid)
    return action


def confirm(pid: int, action_id: str | None = None) -> dict:
    """The user said go: start the proposed run and wake the Tuner with the outcome."""
    from slm.tuner.session import tuner

    action = _take(pid, action_id)
    try:
        result = _execute(pid, action)
    except Exception as e:
        tuner.send(
            pid,
            f"[Confirmed] The user approved “{action['title']}”, but it couldn't start: {e}. "
            "Explain briefly and propose a fix.",
            role="event",
            meta={"confirmed": action["kind"], "error": True},
        )
        raise
    tuner.send(
        pid,
        f"[Confirmed] The user approved “{action['title']}”. {result['text']} "
        "Tell them in a line, then carry on with work that doesn't need a run while it goes.",
        role="event",
        meta={"confirmed": action["kind"], "job_id": result.get("job_id")},
    )
    return result


def decline(pid: int, action_id: str | None = None, reason: str = "") -> dict:
    from slm.tuner.session import tuner

    action = _take(pid, action_id)
    why = f" They said: “{reason.strip()[:400]}”." if reason.strip() else ""
    tuner.send(
        pid,
        f"[Declined] The user said “not now” to “{action['title']}”.{why} Don't propose the same thing again "
        "unchanged: adjust it to what they said, or ask one short question about what they'd prefer.",
        role="event",
        meta={"declined": action["kind"]},
    )
    return {"declined": action["id"]}


def _execute(pid: int, action: dict) -> dict:
    from slm.tuner.tools import _submit

    kind, payload = action["kind"], action["payload"]
    with Session(engine()) as s:
        st = studio_state(s, pid)
        st.stage = STAGE.get(kind, st.stage)
        if kind in SPEND_TOOL:
            # One call, with exactly these arguments. Approving a spend does not reopen a round:
            # the next spend needs its own card.
            st.granted = [*st.granted, {"tool": SPEND_TOOL[kind], "args": payload.get("args", {})}]
        else:
            # A finished project isn't closed: a confirmed run reopens it (autopilot carries the round
            # through to its next proposal), and the next export completes it again.
            st.completed, st.autopilot, st.stalled_nudges = False, True, 0
        s.add(st)
        s.commit()
    if kind in SPEND_TOOL:
        canvas_changed(pid)
        return {"text": f"Call {SPEND_TOOL[kind]} now, with these arguments: {json.dumps(payload.get('args', {}))}."}
    if kind == "model":
        return apply_base_model(pid, payload["repo_id"])
    job = _submit(kind, payload["config"], pid)
    out = {"job_id": job.id, "text": f"{kind} job {job.id} started; you'll get a job update when it finishes."}
    ahead = overview()["capacity"]["gpu_running"] if kind in ("sft", "dpo") else None
    if ahead and ahead["job_id"] != job.id:
        out["text"] = (
            f"{kind} job {job.id} is queued behind {ahead['label']} for {ahead['project']}; "
            "you'll get a job update when it finishes."
        )
    canvas_changed(pid)
    return out


def apply_base_model(pid: int, repo_id: str) -> dict:
    """Set the project's base model, downloading it unless it's already on this Mac."""
    from slm.tuner.tools import _submit

    with Session(engine()) as s:
        p = s.get(Project, pid)
        if base_model_locked(s, p, repo_id):
            raise ValueError(SWITCH_BASE)
        p.base_model = repo_id
        p.current_adapter_path = None
        local = manage.local_path_for(repo_id)
        if local is None and (rec := manage.register_local(repo_id)) is not None:
            local = rec.local_path  # already in the HF cache: no download needed
        p.current_model_path = local
        s.add(p)
        s.commit()
    canvas_changed(pid)
    if local:
        return {"text": f"{repo_id} is already on this Mac, so it's ready to use now."}
    job = _submit("download", {"repo_id": repo_id}, pid)
    return {"job_id": job.id, "text": f"Downloading {repo_id} (job {job.id}); you'll get a job update when it's done."}
