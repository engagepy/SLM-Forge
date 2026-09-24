"""Run Tuner turns in the background and stream them to the Studio.

One turn runs at a time per project; messages that arrive meanwhile (from the user, a finished
job, or completed comparisons) queue up and run as the next turn. Everything the UI needs is
published on the bus topic `tuner:{project_id}`:

    turn_start · delta {text} · tool_start {name, args} · tool_end {name, output}
    message {message} · turn_end · canvas · error {message}
"""

import asyncio
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from agents import Runner, SQLiteSession
from agents.exceptions import MaxTurnsExceeded
from sqlmodel import Session, select

from slm.agents.provider import OpenAIProvider
from slm.config import get_settings
from slm.db import (
    Checkpoint,
    Dataset,
    DatasetVersion,
    Feedback,
    Job,
    SftExample,
    TunerMessage,
    count,
    engine,
    studio_state,
)
from slm.events import bus, canvas_changed, shutting_down
from slm.tuner.agent import build_agent
from slm.tuner.tools import TunerContext


def _publish(pid: int, event: dict) -> None:
    bus.publish(f"tuner:{pid}", event)


def _busy_changed(pid: int, busy: bool) -> None:
    """Announce on the machine-wide feed when a project's Tuner starts or stops thinking, so the
    sessions list updates by push instead of polling."""
    bus.publish("jobs", {"type": "tuner", "project_id": pid, "busy": busy})


def save_message(pid: int, role: str, content: str, meta: dict | None = None) -> TunerMessage:
    with Session(engine()) as s:
        m = TunerMessage(project_id=pid, role=role, content=content, meta=meta or {})
        s.add(m)
        s.commit()
        s.refresh(m)
    _publish(pid, {"type": "message", "message": m.model_dump(mode="json")})
    return m


def _update_message(msg_id: int, meta: dict) -> TunerMessage:
    with Session(engine()) as s:
        m = s.get(TunerMessage, msg_id)
        m.meta = m.meta | meta
        s.add(m)
        s.commit()
        s.refresh(m)
        return m


class Tuner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy: set[int] = set()
        self._pending: dict[int, list[str]] = {}
        self.model_override = None  # tests inject a scripted model here
        # One long-lived loop for every turn: the Agents SDK's shared OpenAI client binds to the
        # loop it first runs on, so a fresh asyncio.run() per turn fails with "Event loop is closed".
        self._loop: asyncio.AbstractEventLoop | None = None
        # Tools run here (asyncio.to_thread). Owned, so shutdown can stop it without joining a tool
        # that's still waiting on something.
        self._tools = ThreadPoolExecutor(thread_name_prefix="tuner-tools")
        self._runs: dict[int, object] = {}  # project → the streaming run in progress
        self._halted: set[int] = set()  # projects the user stopped: no new work until they speak

    def halt(self, pid: int) -> None:
        """Stop this project's agent now: cancel the turn in progress and drop queued messages.
        Tools refuse to start jobs until the user sends a message or resumes."""
        with self._lock:
            self._halted.add(pid)
            self._pending.pop(pid, None)
            run = self._runs.get(pid)
        if run is not None and self._loop is not None:
            # "immediate": "after_turn" would still execute pending tool calls (e.g. start training).
            self._loop.call_soon_threadsafe(run.cancel, "immediate")

    def unhalt(self, pid: int) -> None:
        with self._lock:
            self._halted.discard(pid)

    def is_halted(self, pid: int) -> bool:
        return pid in self._halted

    def _event_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._loop.set_default_executor(self._tools)
                threading.Thread(target=self._loop.run_forever, daemon=True, name="tuner-loop").start()
            return self._loop

    def shutdown(self) -> None:
        """Stop every turn and let tool threads finish: the server is going down."""
        shutting_down.set()
        with self._lock:
            runs, loop = list(self._runs.values()), self._loop
            self._pending.clear()
        if loop is not None:
            for run in runs:
                loop.call_soon_threadsafe(run.cancel, "immediate")
        # Tool threads return within about a second (their waits watch `shutting_down`); give them
        # that, so their tasks complete, then stop the loop. Anything slower is abandoned, not joined.
        self._tools.shutdown(wait=False, cancel_futures=True)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(t.name.startswith("tuner-tools") for t in threading.enumerate()):
            time.sleep(0.05)
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)

    def is_busy(self, pid: int) -> bool:
        return pid in self._busy

    def send(self, pid: int, text: str, role: str = "user", meta: dict | None = None) -> None:
        """Record a message and make sure a turn will process it."""
        save_message(pid, role, text, meta)
        if role == "user":
            self.unhalt(pid)  # the user speaking is permission to act again
            reset_stall(pid)  # and counts as fresh direction
        elif self.is_halted(pid):
            return  # stopped: record the event, but don't run the agent
        with self._lock:
            self._pending.setdefault(pid, []).append(text)
            if pid in self._busy:
                return  # the running drain picks it up next
            self._busy.add(pid)
        _busy_changed(pid, True)
        asyncio.run_coroutine_threadsafe(self._drain(pid), self._event_loop())

    async def _drain(self, pid: int) -> None:
        """Run turns until nothing is pending (messages that arrive mid-turn batch into the next)."""
        try:
            while True:
                with self._lock:
                    batch = self._pending.pop(pid, [])
                if not batch:
                    # Nothing queued: autopilot decides whether the Tuner should keep going.
                    nudge = await asyncio.to_thread(autopilot_nudge, pid)
                    with self._lock:
                        batch = self._pending.pop(pid, [])
                        if nudge:
                            batch.append(nudge)
                        if not batch:
                            self._busy.discard(pid)
                            _busy_changed(pid, False)
                            return
                await self._turn(pid, "\n\n".join(batch))
        except Exception:
            traceback.print_exc()
            with self._lock:
                self._busy.discard(pid)
            _busy_changed(pid, False)

    async def _turn(self, pid: int, text: str) -> None:
        _publish(pid, {"type": "turn_start"})
        settings = get_settings()
        segment: list[str] = []
        tool_msgs: dict[str, int] = {}

        def flush() -> None:
            content = "".join(segment).strip()
            segment.clear()
            if content:
                save_message(pid, "assistant", content)

        try:
            if self.model_override is None:
                OpenAIProvider()  # configures the SDK's key and tracing, or raises a clear error
            agent = build_agent(self.model_override)
            session = SQLiteSession(f"project-{pid}", settings.workspace / "tuner_sessions.db")
            result = Runner.run_streamed(agent, text, context=TunerContext(pid), session=session, max_turns=40)
            self._runs[pid] = result
            async for ev in result.stream_events():
                if ev.type == "raw_response_event" and getattr(ev.data, "type", "") == "response.output_text.delta":
                    segment.append(ev.data.delta)
                    _publish(pid, {"type": "delta", "text": ev.data.delta})
                elif ev.type == "run_item_stream_event" and ev.name == "tool_called":
                    flush()  # text before a tool call is its own message
                    raw = ev.item.raw_item
                    name = getattr(raw, "name", "tool")
                    try:
                        args = json.loads(getattr(raw, "arguments", "") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    m = save_message(pid, "tool", name, {"name": name, "args": args, "status": "running"})
                    tool_msgs[getattr(raw, "call_id", name)] = m.id
                    _publish(pid, {"type": "tool_start", "name": name, "args": args, "message_id": m.id})
                elif ev.type == "run_item_stream_event" and ev.name == "tool_output":
                    raw = ev.item.raw_item
                    call_id = raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)
                    output = str(ev.item.output)
                    if (mid := tool_msgs.get(call_id)) is not None:
                        m = _update_message(mid, {"status": "done", "output": output[:4000]})
                        _publish(pid, {"type": "message", "message": m.model_dump(mode="json")})
                    _publish(pid, {"type": "tool_end", "output": output[:500]})
            flush()
        except MaxTurnsExceeded:
            flush()
            save_message(pid, "event", "The Tuner paused after many steps. Say “continue” to carry on.")
        except Exception as e:
            flush()
            traceback.print_exc()
            save_message(pid, "event", f"The Tuner hit an error: {e.__class__.__name__}: {e}", {"error": True})
            _publish(pid, {"type": "error", "message": str(e)})
        finally:
            self._runs.pop(pid, None)
            _publish(pid, {"type": "turn_end"})
            canvas_changed(pid)


tuner = Tuner()

AUTOPILOT_TEXT = (
    "[Autopilot] Keep going with the next step. Do the preparation yourself; when the next step is a "
    "run (base model, training, export), propose it for the user to confirm and stop there. "
    "If the model is exported and you're done, call finish_project."
)
MAX_STALLED_NUDGES = 2


def _progress_signature(s: Session, pid: int) -> list[int]:
    """Counts that change whenever real work happens."""
    return [count(s, m, m.project_id == pid) for m in (Job, Dataset, DatasetVersion, Checkpoint, Feedback, SftExample)]


_last_signature: dict[int, list[int]] = {}


def reset_stall(pid: int) -> None:
    _last_signature.pop(pid, None)
    with Session(engine()) as s:
        if (st := studio_state(s, pid)).stalled_nudges:
            st.stalled_nudges = 0
            s.add(st)
            s.commit()


def autopilot_nudge(pid: int) -> str | None:
    """After a turn: should the Tuner keep going on its own? Returns the nudge text, or None.

    Not when autopilot is off or the project is finished, not while a job runs (its completion
    wakes the Tuner), and not while a proposal or comparisons wait for the user. Pauses itself after
    MAX_STALLED_NUDGES nudges in a row that produced no progress, so it can't spin."""
    with Session(engine()) as s:
        st = studio_state(s, pid)
        if not st.autopilot or st.completed:
            return None
        active = s.exec(select(Job.id).where(Job.project_id == pid, Job.status.in_(["queued", "running"]))).first()
        if active is not None:
            return None
        if any(c.get("status") == "pending" for c in st.comparisons):
            return None
        if st.pending_action:
            return None  # waiting for the user to confirm a run: their decision wakes the Tuner
        sig = _progress_signature(s, pid)
        if _last_signature.get(pid) == sig:
            st.stalled_nudges += 1
        else:
            st.stalled_nudges = 0
        _last_signature[pid] = sig
        paused = st.stalled_nudges >= MAX_STALLED_NUDGES
        if paused:
            st.autopilot, st.stalled_nudges = False, 0
        s.add(st)
        s.commit()
    if paused:
        save_message(
            pid,
            "event",
            "Autopilot paused: the Tuner went two steps without making progress. "
            "Tell it what to do, or switch autopilot back on.",
            {"autopilot_paused": True},
        )
        return None
    save_message(pid, "event", AUTOPILOT_TEXT, {"autopilot": True})
    return AUTOPILOT_TEXT


def mark_completed(pid: int) -> None:
    """A successful export means the model is delivered, so the project is complete and autopilot
    stops nudging. This doesn't rely on the Tuner remembering to call finish_project; it's still
    woken to write its wrap-up. Confirming a new run (or switching autopilot back on) reopens it."""
    with Session(engine()) as s:
        st = studio_state(s, pid)
        st.completed, st.stage = True, "export"
        s.add(st)
        s.commit()
    canvas_changed(pid)


def job_update_text(job: Job, instruct: bool = True) -> str:
    r = job.result or {}
    parts = [f"[Job update] {job.kind} job {job.id} {job.status}."]
    if job.error:
        parts.append(f"Error: {job.error[:400]}")
    for key in ("metrics", "warnings", "path", "size_gb", "min_ram_gb", "local_path"):
        if key in r:
            parts.append(f"{key}: {json.dumps(r[key], default=str)[:600]}")
    if instruct:
        parts.append(
            "Explain what this means for the user and continue with the next step (propose the next run "
            "for them to confirm rather than expecting it to start by itself)."
        )
    return " ".join(parts)


def on_job_finished(job: Job) -> None:
    """Wake the Tuner when a long job it started finishes (it awaits short ones itself)."""
    if job.project_id is None:
        return
    if job.kind == "export" and job.status == "succeeded":
        mark_completed(job.project_id)
    if not (job.config or {}).get("notify") or job.status == "cancelled":
        return  # a job the user cancelled must not wake the agent (it would start new work)
    with Session(engine()) as s:
        autopilot = studio_state(s, job.project_id).autopilot
    if not autopilot:
        # Paused: note the result in the transcript, but only act when the user asks.
        save_message(job.project_id, "event", job_update_text(job, instruct=False), {"job_id": job.id, "quiet": True})
        return
    tuner.send(
        job.project_id,
        job_update_text(job),
        role="event",
        meta={"job_id": job.id, "kind": job.kind, "status": job.status},
    )
