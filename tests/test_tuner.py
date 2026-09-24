"""The Tuner's tools, invoked exactly as the Agents SDK invokes them, and the Studio API."""

import asyncio
import json

import pytest
from agents.tool_context import ToolContext
from fastapi.testclient import TestClient
from sqlmodel import select

from slm.db import Dataset, Job, PreferencePair, Project, SftExample, StudioState, TunerMessage
from slm.tuner import tools
from slm.tuner.session import job_update_text, on_job_finished, tuner
from slm.tuner.tools import ALL_TOOLS, TunerContext

TOOLS = {t.name: t for t in ALL_TOOLS}


def call(project_id: int, tool_name: str, /, **args):
    """Invoke a tool through the SDK's own entry point (schema parsing, context, thread offload)."""
    tool = TOOLS[tool_name]
    ctx = ToolContext(
        context=TunerContext(project_id), tool_name=tool_name, tool_call_id="t1", tool_arguments=json.dumps(args)
    )
    out = asyncio.run(tool.on_invoke_tool(ctx, json.dumps(args)))
    return out


def test_every_tool_has_a_schema_without_the_context_param():
    for t in ALL_TOOLS:
        assert "ctx" not in t.params_json_schema.get("properties", {}), t.name
        assert t.description, t.name


def test_update_project_and_set_stage(session, project):
    call(
        project.id,
        "update_project",
        name="Cooking helper",
        goal="Answer cooking questions",
        system_prompt="Be concise.",
    )
    call(project.id, "set_stage", stage="data", note="Looking for recipes")
    session.expire_all()
    p = session.get(Project, project.id)
    st = session.get(StudioState, project.id)
    assert (p.name, p.system_prompt) == ("Cooking helper", "Be concise.")
    assert (st.stage, st.note) == ("data", "Looking for recipes")


def test_invalid_stage_is_reported_to_the_model_not_raised(project):
    out = call(project.id, "set_stage", stage="launch", note="x")
    assert "stage must be one of" in str(out)


def test_get_status_describes_the_project(project):
    out = call(project.id, "get_status")
    assert out["project"]["goal"] == project.goal
    assert out["hardware"]["memory_gb"] > 0
    assert out["feedback"]["judgements"] == 0


def test_prepare_dataset_rejects_a_mapping_that_maps_nothing(session, project, tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps({"title": f"t{i}", "method": ["a", "b"]}) for i in range(5)))
    ds = Dataset(project_id=project.id, name="r", source="upload", raw_path=str(raw), columns=["title", "method"])
    session.add(ds)
    session.commit()
    out = call(
        project.id,
        "prepare_dataset",
        dataset_id=ds.id,
        format="instruction",
        prompt_column="name",
        response_column="method",
    )
    assert out["status"] == "mapping doesn't work" and "not in the data" in out["errors"][0]
    assert not session.exec(select(Job)).all()  # nothing was queued


def test_review_uses_unambiguous_ids(session, project):
    # A pair and an SFT example can share the same numeric id; "p1" and "s1" must not collide.
    session.add(PreferencePair(project_id=project.id, prompt="q", chosen="good", rejected="bad", approved=False))
    session.add(SftExample(project_id=project.id, messages=[{"role": "user", "content": "q"}], approved=False))
    session.commit()
    pair_id = session.exec(select(PreferencePair)).one().id
    ex_id = session.exec(select(SftExample)).one().id
    assert pair_id == ex_id == 1
    out = call(project.id, "review_synthetic_examples", approve_ids=[f"s{ex_id}"])
    session.expire_all()
    assert session.get(SftExample, ex_id).approved and not session.get(PreferencePair, pair_id).approved
    assert out["still_pending"] == 1
    call(project.id, "review_synthetic_examples", reject_ids=[f"p{pair_id}"], approve_all=True)
    session.expire_all()
    assert session.get(PreferencePair, pair_id) is None  # rejected wins over approve_all


def test_long_jobs_wake_the_tuner_and_short_ones_do_not(session, project, monkeypatch):
    sent = []
    monkeypatch.setattr(tuner, "send", lambda pid, text, role="user", meta=None: sent.append((pid, role, text)))
    done = Job(id=5, project_id=project.id, kind="sft", status="succeeded", config={"notify": True},
               result={"metrics": {"val_loss": 1.2}, "warnings": [{"code": "diverged"}]})  # fmt: skip
    on_job_finished(done)
    on_job_finished(Job(id=6, project_id=project.id, kind="prepare_dataset", status="succeeded", config={}))
    assert len(sent) == 1 and sent[0][1] == "event"
    assert "sft job 5 succeeded" in sent[0][2] and "diverged" in sent[0][2]
    assert "val_loss" in job_update_text(done)


@pytest.fixture
def client():
    from slm.api.app import app

    return TestClient(app)


def test_studio_snapshot_and_kickoff(client, project, monkeypatch):
    sent = []
    monkeypatch.setattr(tuner, "send", lambda pid, text, role="user", meta=None: sent.append(role))
    snap = client.get(f"/api/projects/{project.id}/studio").json()
    assert snap["stage"] == "goal" and snap["tuner_busy"] is False and snap["model"] is None
    assert client.post(f"/api/projects/{project.id}/tuner/start").json() == {"started": True}
    assert sent == ["event"]


def test_judging_the_last_comparison_records_feedback_and_wakes_the_tuner(client, session, project, monkeypatch):
    sent = []
    monkeypatch.setattr(tuner, "send", lambda pid, text, role="user", meta=None: sent.append(text))
    session.add(StudioState(project_id=project.id, comparisons=[
        {"id": "c1", "prompt": "Boil an egg?", "a": "6 minutes.", "b": "Idk.", "status": "pending"},
        {"id": "c2", "prompt": "Steak temp?", "a": "55°C", "b": "55°C", "status": "pending"},
    ]))  # fmt: skip
    session.commit()
    r = client.post(f"/api/projects/{project.id}/studio/judge", json={"comparison_id": "c1", "choice": "a"}).json()
    assert r["preference_pairs"] == 1 and r["batch_done"] is False and not sent
    r = client.post(
        f"/api/projects/{project.id}/studio/judge",
        json={
            "comparison_id": "c2",
            "choice": "both_bad",
            "edited_answer": "Pull at 52°C, rest to 55°C.",
            "critique": "no resting",
        },
    ).json()
    assert r["batch_done"] is True and r["sft_examples"] == 1
    assert len(sent) == 1 and "no resting" in sent[0]
    assert (
        client.post(f"/api/projects/{project.id}/studio/judge", json={"comparison_id": "c1", "choice": "b"}).status_code
        == 404
    )


def test_messages_endpoint_returns_transcript(client, session, project):
    session.add(TunerMessage(project_id=project.id, role="assistant", content="Hi"))
    session.commit()
    assert [m["content"] for m in client.get(f"/api/projects/{project.id}/tuner/messages").json()] == ["Hi"]


def test_canvas_changed_publishes(project):
    got = []
    from slm.events import bus

    async def listen():
        async with bus.subscribe(f"tuner:{project.id}") as q:
            tools.canvas_changed(project.id)
            got.append(await asyncio.wait_for(q.get(), 1))

    asyncio.run(listen())
    assert got == [{"type": "canvas"}]


# ── autopilot ────────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_autopilot(project):
    from slm.tuner import session

    session._last_signature.clear()
    return project


def test_autopilot_nudges_when_idle(session, fresh_autopilot):
    from slm.tuner.session import AUTOPILOT_TEXT, autopilot_nudge

    session.add(StudioState(project_id=fresh_autopilot.id))
    session.commit()
    assert autopilot_nudge(fresh_autopilot.id) == AUTOPILOT_TEXT
    # The nudge is recorded in the transcript but flagged so the chat can hide it.
    msg = session.exec(select(TunerMessage)).all()[-1]
    assert msg.meta == {"autopilot": True}


@pytest.mark.parametrize(
    "setup",
    ["off", "completed", "job_running", "user_comparing"],
)
def test_autopilot_holds_off(session, fresh_autopilot, setup):
    from slm.tuner.session import autopilot_nudge

    pid = fresh_autopilot.id
    st = StudioState(project_id=pid)
    if setup == "off":
        st.autopilot = False
    elif setup == "completed":
        st.completed = True
    elif setup == "user_comparing":
        st.comparisons = [{"id": "c1", "status": "pending"}]
    else:
        session.add(Job(project_id=pid, kind="sft", status="running"))
    session.add(st)
    session.commit()
    assert autopilot_nudge(pid) is None


def test_autopilot_pauses_after_nudges_without_progress_and_resets_on_progress(session, fresh_autopilot):
    from slm.tuner.session import autopilot_nudge, reset_stall

    pid = fresh_autopilot.id
    session.add(StudioState(project_id=pid))
    session.commit()
    assert autopilot_nudge(pid)  # first nudge sets the baseline
    session.add(Job(project_id=pid, kind="prepare_dataset", status="succeeded"))  # progress
    session.commit()
    assert autopilot_nudge(pid)  # progress → keep going, stall counter reset
    assert autopilot_nudge(pid)  # no progress: stall 1
    assert autopilot_nudge(pid) is None  # no progress again: stall 2 → paused
    session.expire_all()
    st = session.get(StudioState, pid)
    assert st.autopilot is False
    paused = session.exec(select(TunerMessage)).all()[-1]
    assert paused.meta == {"autopilot_paused": True}
    # Switching it back on (or the user steering) starts clean.
    st.autopilot = True
    session.add(st)
    session.commit()
    reset_stall(pid)
    assert autopilot_nudge(pid)


def test_autopilot_endpoint_toggles_and_wakes(client, session, project, monkeypatch):
    sent = []
    monkeypatch.setattr(tuner, "send", lambda pid, text, role="user", meta=None: sent.append(text))
    monkeypatch.setattr(tuner, "is_busy", lambda pid: False)
    assert client.post(f"/api/projects/{project.id}/studio/autopilot", json={"on": False}).json() == {
        "autopilot": False
    }
    assert not sent
    assert client.post(f"/api/projects/{project.id}/studio/autopilot", json={"on": True}).json() == {"autopilot": True}
    assert len(sent) == 1
    snap = client.get(f"/api/projects/{project.id}/studio").json()
    assert snap["autopilot"] is True and snap["completed"] is False


def test_find_base_models_puts_local_models_first(project, monkeypatch):
    from types import SimpleNamespace

    from slm.models import hub, manage

    monkeypatch.setattr(
        manage,
        "local_models",
        lambda: [
            {"repo_id": "org/tiny-0.5B-4bit", "params_b": 0.5, "bits": 4.0, "train_memory_gb": 2.4, "fit": "fits"}
        ],
    )
    remote = SimpleNamespace(
        id="org/other-1.5B", params=1.5e9, bits=4.0, train_estimate_gb=3.2, fit="fits", downloads=9
    )
    monkeypatch.setattr(hub, "search_models", lambda q, max_params_b=None, limit=12: [remote])
    out = call(project.id, "find_base_models", query="tiny instruct")
    assert [m["repo_id"] for m in out] == ["org/tiny-0.5B-4bit", "org/other-1.5B"]
    assert out[0]["already_on_this_mac"] is True and out[1]["already_on_this_mac"] is False


def test_every_defined_tool_is_registered():
    # Regression: two tools were once defined with @tool but never added to ALL_TOOLS, so the
    # agent silently couldn't use them.
    import ast
    import pathlib

    src = pathlib.Path(tools.__file__).read_text()
    defined = {
        n.name
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and any(getattr(d, "id", None) == "tool" for d in n.decorator_list)
    }
    assert defined == {t.name for t in ALL_TOOLS}
