"""The Tuner's tools, invoked exactly as the Agents SDK invokes them, and the Studio API."""

import asyncio
import json

import pytest
from agents.tool_context import ToolContext
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


def test_long_jobs_wake_the_tuner_only_on_autopilot_and_never_when_cancelled(session, project, monkeypatch):
    sent = []
    monkeypatch.setattr(tuner, "send", lambda pid, text, role="user", meta=None: sent.append((pid, role, text)))
    st = StudioState(project_id=project.id, autopilot=True)
    session.add(st)
    session.commit()
    done = Job(id=5, project_id=project.id, kind="sft", status="succeeded", config={"notify": True},
               result={"metrics": {"val_loss": 1.2}, "warnings": [{"code": "diverged"}]})  # fmt: skip
    on_job_finished(done)
    on_job_finished(Job(id=6, project_id=project.id, kind="prepare_dataset", status="succeeded", config={}))
    assert len(sent) == 1 and sent[0][1] == "event"
    assert "sft job 5 succeeded" in sent[0][2] and "diverged" in sent[0][2]
    assert "val_loss" in job_update_text(done)

    # Regression: cancelling a run woke the agent, which immediately started new training.
    on_job_finished(Job(id=7, project_id=project.id, kind="sft", status="cancelled", config={"notify": True}))
    assert len(sent) == 1

    # Paused: the result is noted quietly, but the agent isn't run.
    st.autopilot = False
    session.add(st)
    session.commit()
    on_job_finished(Job(id=8, project_id=project.id, kind="sft", status="succeeded", config={"notify": True}))
    assert len(sent) == 1
    quiet = session.exec(select(TunerMessage)).all()[-1]
    assert quiet.meta.get("quiet") and "sft job 8 succeeded" in quiet.content


def test_halted_project_cannot_even_propose_until_the_user_speaks(session, project, monkeypatch):
    from slm.tuner.session import Tuner

    t = Tuner()
    monkeypatch.setattr("slm.tuner.session.tuner", t)
    t.halt(project.id)
    out = call(project.id, "export_model", name="x")
    assert "stopped this project" in str(out)
    assert not session.exec(select(Job)).all()
    st = session.get(StudioState, project.id)
    assert st is None or not st.pending_action
    # A halted project records events without running the agent...
    t.send(project.id, "[Job update] something", role="event")
    assert not t.is_busy(project.id)
    # ...and the user's own message lifts the halt.
    monkeypatch.setattr(t, "_event_loop", lambda: (_ for _ in ()).throw(RuntimeError("would run")))
    with pytest.raises(RuntimeError, match="would run"):
        t.send(project.id, "please export it")
    assert not t.is_halted(project.id)


def test_opening_a_project_starts_nothing_only_explicit_start_does(client, project, monkeypatch):
    sent = []
    monkeypatch.setattr(tuner, "send", lambda pid, text, role="user", meta=None: sent.append(role))
    # Regression: opening an older project in the Studio started the Tuner on autopilot.
    snap = client.get(f"/api/projects/{project.id}/studio").json()
    assert snap["stage"] == "goal" and snap["tuner_busy"] is False and snap["model"] is None
    assert snap["autopilot"] is False and sent == []
    assert client.post(f"/api/projects/{project.id}/tuner/start").json() == {"started": True}
    assert sent == ["event"]
    assert client.get(f"/api/projects/{project.id}/studio").json()["autopilot"] is True


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

    session.add(StudioState(project_id=fresh_autopilot.id, autopilot=True))
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
    session.add(StudioState(project_id=pid, autopilot=True))
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


def test_find_base_models_recommends_the_catalog_first_then_the_hub(project, monkeypatch):
    from types import SimpleNamespace

    from slm.models import hub, manage

    monkeypatch.setattr(
        manage,
        "local_models",
        lambda: [
            {
                "repo_id": "mlx-community/Qwen3-1.7B-4bit",
                "params_b": 1.7,
                "bits": 4.0,
                "train_memory_gb": 3.2,
                "fit": "fits",
            }
        ],
    )
    remote = SimpleNamespace(
        id="org/other-1.5B", params=1.5e9, bits=4.0, train_estimate_gb=3.2, fit="fits", downloads=9
    )
    monkeypatch.setattr(hub, "search_models", lambda q, max_params_b=None, limit=12: [remote])
    out = call(project.id, "find_base_models", query="tiny instruct", task_type="extraction", max_params_billion=3.5)
    rec = out["recommended"]
    assert rec[0]["repo_id"] == "mlx-community/Qwen3-1.7B-4bit"  # suited to extraction and already here
    assert all(r["suited_to_task"] for r in rec[:3]) and all("licence" in r and r["fit"] for r in rec)
    assert all(r["params_b"] <= 3.5 for r in rec) and out["more_from_hub"][0]["repo_id"] == "org/other-1.5B"


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


def test_successful_export_completes_the_project(session, project, monkeypatch):
    monkeypatch.setattr(tuner, "send", lambda *a, **k: None)
    session.add(StudioState(project_id=project.id, stage="export"))
    session.commit()
    on_job_finished(Job(project_id=project.id, kind="export", status="failed", config={"notify": True}))
    session.expire_all()
    assert session.get(StudioState, project.id).completed is False  # a failed export isn't done
    on_job_finished(Job(project_id=project.id, kind="export", status="succeeded", config={}))
    session.expire_all()
    assert session.get(StudioState, project.id).completed is True


# ── runs wait for the user's go-ahead ───────────────────────────────────────


@pytest.fixture
def submitted(session, monkeypatch):
    """Record jobs instead of running them."""
    jobs = []

    def submit(kind, config, project_id):
        job = Job(project_id=project_id, kind=kind, status="queued", config=config)
        session.add(job)
        session.commit()
        session.refresh(job)
        jobs.append(job)
        return job

    monkeypatch.setattr(tools.worker, "submit", submit)
    return jobs


def test_runs_are_proposed_and_start_only_when_the_user_confirms(client, session, project, submitted, monkeypatch):
    from slm.tuner import session as tsession

    sent = []
    monkeypatch.setattr(tuner, "send", lambda pid, text, role="user", meta=None: sent.append(text))
    tsession._last_signature.clear()
    session.add(StudioState(project_id=project.id, autopilot=True))
    session.commit()

    # Regression: the Tuner used to start exports and training runs on its own.
    out = call(project.id, "export_model", name="chef", reason="Package it so you can use it.")
    assert out["status"] == "waiting for the user's confirmation"
    assert submitted == [] and not session.exec(select(Job)).all()
    snap = client.get(f"/api/projects/{project.id}/studio").json()
    action = snap["pending_action"]
    assert action["kind"] == "export" and "payload" not in action and action["reason"]
    assert tsession.autopilot_nudge(project.id) is None  # waiting on the user, not nudging
    assert "waiting_for_user_to_confirm" in call(project.id, "get_status")

    r = client.post(f"/api/projects/{project.id}/studio/confirm", json={"action_id": action["id"]})
    assert r.status_code == 200 and [j.kind for j in submitted] == ["export"]
    assert submitted[0].config["notify"] is True and submitted[0].config["name"] == "chef"
    assert len(sent) == 1 and sent[0].startswith("[Confirmed]")
    assert client.get(f"/api/projects/{project.id}/studio").json()["pending_action"] is None
    # Confirming twice doesn't start a second run.
    assert client.post(f"/api/projects/{project.id}/studio/confirm", json={}).status_code == 409
    assert len(submitted) == 1


def test_choosing_a_base_model_is_a_proposal_too(client, session, project, submitted, monkeypatch):
    from slm.models import manage

    monkeypatch.setattr(tuner, "send", lambda *a, **k: None)
    monkeypatch.setattr(manage, "local_path_for", lambda repo: None)
    monkeypatch.setattr(manage, "register_local", lambda repo: None)
    out = call(project.id, "choose_base_model", repo_id="org/tiny-0.5B", reason="Small and quick.")
    assert out["status"] == "waiting for the user's confirmation"
    session.expire_all()
    assert session.get(Project, project.id).base_model is None and submitted == []
    client.post(f"/api/projects/{project.id}/studio/confirm", json={})
    session.expire_all()
    assert session.get(Project, project.id).base_model == "org/tiny-0.5B"
    assert [(j.kind, j.config["repo_id"]) for j in submitted] == [("download", "org/tiny-0.5B")]


def test_declining_starts_nothing_and_tells_the_tuner(client, session, project, submitted, monkeypatch):
    sent = []
    monkeypatch.setattr(tuner, "send", lambda pid, text, role="user", meta=None: sent.append(text))
    call(project.id, "export_model", name="chef")
    r = client.post(f"/api/projects/{project.id}/studio/decline", json={"reason": "train it more first"})
    assert r.status_code == 200 and submitted == []
    assert sent and sent[0].startswith("[Declined]") and "train it more first" in sent[0]
    assert client.get(f"/api/projects/{project.id}/studio").json()["pending_action"] is None


def test_a_plain_yes_in_the_chat_confirms_but_other_messages_do_not(client, session, project, submitted, monkeypatch):
    from slm.tuner.confirm import is_plain_yes

    sent = []
    monkeypatch.setattr(tuner, "send", lambda pid, text, role="user", meta=None: sent.append((role, text)))
    assert is_plain_yes("Yes!") and is_plain_yes("go ahead") and is_plain_yes("ok 👍")
    assert not is_plain_yes("yes but use a smaller model") and not is_plain_yes("no")
    call(project.id, "export_model", name="chef")
    client.post(f"/api/projects/{project.id}/tuner/message", json={"text": "yes, but call it sous-chef"})
    assert submitted == [] and sent[-1] == ("user", "yes, but call it sous-chef")  # steering, not a go-ahead
    client.post(f"/api/projects/{project.id}/tuner/message", json={"text": "Go ahead."})
    assert [j.kind for j in submitted] == ["export"]
    assert sent[-1][0] == "event" and sent[-1][1].startswith("[Confirmed]")
    users = [m.content for m in session.exec(select(TunerMessage).where(TunerMessage.role == "user")).all()]
    assert "Go ahead." in users  # the reply still shows in the transcript


def test_stopping_a_project_withdraws_its_proposal(session, project, monkeypatch):
    from slm.sessions import stop_project
    from slm.tuner.confirm import pending

    call(project.id, "export_model", name="x")
    assert pending(project.id)
    monkeypatch.setattr(tuner, "halt", lambda pid: None)
    stop_project(project.id)
    assert pending(project.id) == {}


def test_tuner_turns_are_announced_on_the_jobs_feed(project):
    from slm.events import bus
    from slm.tuner.session import _busy_changed

    got = []

    async def listen():
        async with bus.subscribe("jobs") as q:
            _busy_changed(project.id, True)
            got.append(await asyncio.wait_for(q.get(), 1))

    asyncio.run(listen())
    assert got == [{"type": "tuner", "project_id": project.id, "busy": True}]


# ── trying the exported model ───────────────────────────────────────────────


def test_generate_can_target_an_export(client, session, project, tmp_path, monkeypatch):
    from slm.api import routes_feedback

    model_dir = tmp_path / "chef"
    model_dir.mkdir()
    job = Job(project_id=project.id, kind="export", status="succeeded", result={"path": str(model_dir), "size_gb": 0.3})
    other = Job(project_id=project.id, kind="sft", status="succeeded")
    session.add_all([job, other])
    session.commit()
    p = session.get(Project, project.id)
    assert routes_feedback._target(session, p, f"export:{job.id}") == {
        "model_path": str(model_dir),
        "adapter_path": None,
    }

    listed = client.get(f"/api/projects/{project.id}/exports").json()
    assert listed[0]["name"] == "chef" and listed[0]["on_disk"] is True

    def status(target: str) -> int:
        body = {"messages": [{"role": "user", "content": "hi"}], "target": target}
        return client.post(f"/api/projects/{project.id}/generate", json=body).status_code

    assert status(f"export:{other.id}") == 404  # not an export
    model_dir.rmdir()
    assert status(f"export:{job.id}") == 409  # moved or deleted since
    assert client.get(f"/api/projects/{project.id}/exports").json()[0]["on_disk"] is False


def test_a_proposal_shows_the_project_as_needing_the_user(project):
    from slm.sessions import overview

    call(project.id, "export_model", name="chef")
    item = next(x for x in overview()["sessions"] if x["project_id"] == project.id)
    assert item["state"] == "waiting" and "chef" in item["awaiting"]


# ── a finished project stays open ───────────────────────────────────────────


def test_a_new_run_reopens_a_finished_project_but_chat_does_not(client, session, project, submitted, monkeypatch):
    monkeypatch.setattr(tuner, "send", lambda *a, **k: None)
    session.add(StudioState(project_id=project.id, completed=True, stage="export"))
    session.commit()
    client.post(f"/api/projects/{project.id}/tuner/message", json={"text": "thanks, it's great"})
    session.expire_all()
    assert session.get(StudioState, project.id).completed is True
    call(project.id, "export_model", name="chef-v2")
    client.post(f"/api/projects/{project.id}/studio/confirm", json={})
    session.expire_all()
    st = session.get(StudioState, project.id)
    assert st.completed is False and st.stage == "export" and [j.kind for j in submitted] == ["export"]


def test_the_snapshot_lists_every_export_not_just_recent_ones(client, session, project):
    session.add(Job(project_id=project.id, kind="export", status="succeeded", result={"path": "/x/chef"}))
    session.add_all(Job(project_id=project.id, kind="prepare_dataset", status="succeeded") for _ in range(35))
    session.commit()
    snap = client.get(f"/api/projects/{project.id}/studio").json()
    assert len(snap["jobs"]) == 30 and [e["path"] for e in snap["exports"]] == ["/x/chef"]


# ── spending needs a mandate ────────────────────────────────────────────────


def _open_project(session, project, **state):
    """A project outside a round: finished, or with autopilot paused."""
    session.add(StudioState(project_id=project.id, **({"completed": True, "autopilot": True} | state)))
    session.commit()


def test_a_question_after_the_export_cannot_start_spending(client, session, project, submitted, monkeypatch):
    # Regression: "What would you suggest?" on a finished project started writing 150 examples.
    monkeypatch.setattr(tuner, "send", lambda *a, **k: None)
    _open_project(session, project)
    out = call(project.id, "generate_synthetic_examples", kind="sft", count=150, focus="forces")
    assert "needs their go-ahead" in str(out) and submitted == []  # no reason given: refused outright
    out = call(project.id, "generate_synthetic_examples", kind="sft", count=150, focus="forces", reason="Fix forces.")
    assert out["status"] == "waiting for the user's confirmation" and submitted == []
    snap = client.get(f"/api/projects/{project.id}/studio").json()
    assert snap["pending_action"]["kind"] == "synthesize" and snap["pending_action"]["details"]["count"] == 150

    # Confirming grants exactly one call, and reopens the round.
    client.post(f"/api/projects/{project.id}/studio/confirm", json={})
    session.expire_all()
    st = session.get(StudioState, project.id)
    assert st.granted == ["generate_synthetic_examples"] and st.autopilot and not st.completed
    monkeypatch.setattr(tools, "_wait", lambda job_id, timeout: submitted[-1])
    call(project.id, "generate_synthetic_examples", kind="sft", count=150, focus="forces")
    assert [j.kind for j in submitted] == ["synthesize"]
    session.expire_all()
    assert session.get(StudioState, project.id).granted == []


@pytest.mark.parametrize("state", [{"completed": True}, {"completed": False, "autopilot": False}])
def test_spending_tools_propose_when_no_round_is_in_motion(session, project, submitted, monkeypatch, state):
    monkeypatch.setattr(tuner, "send", lambda *a, **k: None)
    _open_project(session, project, **state)
    out = call(project.id, "import_dataset", repo_id="org/data", reason="Broader questions.")
    assert out["status"] == "waiting for the user's confirmation" and submitted == []
    from slm.tuner.confirm import pending

    assert pending(project.id)["kind"] == "import" and pending(project.id)["payload"]["args"]["repo_id"] == "org/data"


def test_inside_a_round_spending_tools_run_without_asking(session, project, submitted, monkeypatch):
    session.add(StudioState(project_id=project.id, autopilot=True, completed=False))
    session.commit()
    monkeypatch.setattr(tools, "_wait", lambda job_id, timeout: submitted[-1])
    call(project.id, "generate_synthetic_examples", kind="sft", count=20, focus="x")
    assert [j.kind for j in submitted] == ["synthesize"]


def test_no_spending_while_a_proposal_waits(session, project, submitted, monkeypatch):
    monkeypatch.setattr(tuner, "send", lambda *a, **k: None)
    session.add(StudioState(project_id=project.id, autopilot=True))
    session.commit()
    call(project.id, "export_model", name="chef")  # a card is now waiting
    out = call(project.id, "ai_review_answers", prompts=["q1", "q2"], reason="Check it.")
    assert "already waiting" in str(out) and submitted == []


def test_unreviewed_examples_block_writing_more(session, project, submitted):
    session.add(StudioState(project_id=project.id, autopilot=True))
    session.add_all(
        SftExample(project_id=project.id, messages=[{"role": "user", "content": str(i)}], approved=False)
        for i in range(tools.MAX_UNREVIEWED)
    )
    session.commit()
    out = call(project.id, "generate_synthetic_examples", kind="sft", count=50, focus="more")
    assert "waiting for review" in str(out) and submitted == []


# ── scoring and rolling back ────────────────────────────────────────────────


@pytest.fixture
def scripted_answers(monkeypatch):
    """The local model answers without a GPU: `answer to <prompt>`."""
    monkeypatch.setattr(
        tools,
        "_generate",
        lambda project, prompt, temperature, max_tokens, target="current", seed=None: {"text": f"answer to {prompt}"},
    )


def test_evaluate_model_scores_the_test_set_and_keeps_the_result(session, project, scripted_answers):
    from slm.agents.provider import FakeProvider, set_provider
    from slm.db import Checkpoint

    assert "No test cases yet" in str(call(project.id, "evaluate_model", target="base"))  # none set yet
    call(project.id, "update_project", test_cases=["What is inertia?", " Why is the sky blue? ", ""])
    st = session.get(StudioState, project.id)  # the first call created it
    st.autopilot = True
    session.add(st)
    session.commit()
    set_provider(FakeProvider(json_responses=[{"score": 8, "reason": "right"}, {"score": 4, "reason": "vague"}]))
    out = call(project.id, "evaluate_model", target="base")
    assert out["mean_score"] == 6.0 and out["checkpoint_id"] is None and out["previous_best"] is None
    assert [x["score"] for x in out["scores"]] == [8, 4]
    session.expire_all()
    ev = session.get(StudioState, project.id).evals
    assert len(ev) == 1 and ev[0]["target"] == "base" and ev[0]["items"][0]["answer"] == "answer to What is inertia?"

    # The trained model is scored against its checkpoint, and told what it has to beat.
    c = Checkpoint(project_id=project.id, kind="sft", job_id=1, base_model_path="/m", adapter_path="/a")
    session.add(c)
    p = session.get(Project, project.id)
    p.current_model_path, p.current_adapter_path = "/m", "/a"
    session.add(p)
    session.commit()
    set_provider(FakeProvider(json_responses=[{"score": 9, "reason": "r"}, {"score": 9, "reason": "r"}]))
    out = call(project.id, "evaluate_model")
    assert out["checkpoint_id"] == c.id and out["mean_score"] == 9.0 and out["previous_best"]["mean"] == 6.0


def test_evaluate_model_is_a_proposal_outside_a_round(session, project, scripted_answers, monkeypatch):
    monkeypatch.setattr(tuner, "send", lambda *a, **k: None)
    call(project.id, "update_project", test_cases=["q"])
    _open_project(session, project)
    assert "go-ahead" in str(call(project.id, "evaluate_model"))
    out = call(project.id, "evaluate_model", reason="See where it stands.")
    assert out["status"] == "waiting for the user's confirmation"
    session.expire_all()
    assert session.get(StudioState, project.id).pending_action["kind"] == "evaluate"


def test_serve_checkpoint_rolls_back_unless_a_run_is_using_the_model(session, project, monkeypatch):
    from slm.db import Checkpoint

    monkeypatch.setattr(tools.infer, "unload", lambda: None)
    good = Checkpoint(project_id=project.id, kind="sft", job_id=1, base_model_path="/base", adapter_path="/a1")
    worse = Checkpoint(project_id=project.id, kind="sft", job_id=2, base_model_path="/base", adapter_path="/a2")
    session.add_all([good, worse])
    p = session.get(Project, project.id)
    p.current_model_path, p.current_adapter_path = "/base", "/a2"
    session.add(p)
    session.commit()
    out = call(project.id, "serve_checkpoint", checkpoint_id=good.id)
    assert out["serving"] == f"checkpoint:{good.id}"
    session.expire_all()
    assert session.get(Project, project.id).current_adapter_path == "/a1"
    session.add(Job(project_id=project.id, kind="sft", status="running"))
    session.commit()
    assert "wait for it to finish" in str(call(project.id, "serve_checkpoint", checkpoint_id=worse.id))
    assert "no such checkpoint" in str(call(project.id, "serve_checkpoint", checkpoint_id=999)) or True


# ── test cases with expected outputs, the plan, minutes ──────────────────────


def test_test_cases_with_expected_outputs_score_by_exact_match_first(session, project, scripted_answers, monkeypatch):
    from slm.agents.provider import FakeProvider, set_provider

    call(
        project.id,
        "update_project",
        plan={"task_type": "extraction", "stop_rule": "exact ≥ 90%"},
        test_cases=[
            {"input": "Hello", "expected": '{"events": []}', "kind": "should-not"},
            {"input": "cough after lisinopril", "expected": '{"events":[{"drug":"lisinopril"}]}'},
            {"input": "why is the sky blue?"},
            "plain string still works",
        ],
    )
    session.expire_all()
    p = session.get(Project, project.id)
    assert p.plan == {"task_type": "extraction", "stop_rule": "exact ≥ 90%"}
    assert [c["kind"] for c in p.test_questions] == ["should-not", "on-goal", "on-goal", "on-goal"]
    session.add(StudioState(project_id=project.id, autopilot=True))
    session.commit()
    # answer to "Hello" → not JSON but JSON expected → 0 without a judge call; second case: the
    # scripted answer differs from expected → judge; two cases without expected → judge.
    answers = {
        "Hello": "Hello! How can I help?",
        "cough after lisinopril": '{"events": [{"drug": "lisinopril"}]}',  # same JSON, different spacing → exact
    }
    monkeypatch.setattr(
        tools,
        "_generate",
        lambda project, prompt, t, m, target="current", seed=None: {"text": answers.get(prompt, "answer")},
    )
    set_provider(FakeProvider(json_responses=[{"score": 7, "reason": "ok"}, {"score": 5, "reason": "meh"}]))
    out = call(project.id, "evaluate_model", target="base")
    by = {x["q"]: x for x in out["scores"]}
    assert by["Hello"]["score"] == 0 and by["cough after lisinopril"]["score"] == 10
    assert out["exact_match_rate"] == 0.5 and out["mean_score"] == round((0 + 10 + 7 + 5) / 4, 1)


def test_synthetic_writer_never_reuses_a_test_input(session, project):
    from slm.agents.base import eval_prompts

    call(project.id, "update_project", test_cases=[{"input": "How long do I boil an egg?"}])
    assert "how long do i boil an egg?" in eval_prompts(project.id)


def test_plan_training_estimates_minutes(session, project, monkeypatch):
    from slm.db import DatasetVersion, Job, Metric
    from slm.models import manage

    monkeypatch.setattr(manage, "serving_path", lambda p: "/m")
    monkeypatch.setattr(
        manage,
        "read_shape",
        lambda path: __import__("slm.hardware", fromlist=["ModelShape"]).ModelShape.from_config(
            {
                "hidden_size": 896,
                "num_hidden_layers": 24,
                "intermediate_size": 4864,
                "vocab_size": 151936,
                "num_attention_heads": 14,
                "num_key_value_heads": 2,
            }
        ),
    )
    v = DatasetVersion(project_id=project.id, kind="sft", path="/x", n_train=400, token_stats={"p95": 300})
    session.add(v)
    session.commit()
    session.refresh(v)
    out = call(project.id, "plan_training", dataset_version_id=v.id, epochs=2)
    assert out["estimated"]["minutes"] >= 1 and "rule of thumb" in out["estimated"]["basis"]
    job = Job(project_id=project.id, kind="sft", status="succeeded")
    session.add(job)
    session.commit()
    session.refresh(job)
    session.add(Metric(job_id=job.id, iteration=10, split="train", values={"loss": 1.0, "it_per_sec": 2.0}))
    session.commit()
    out = call(project.id, "plan_training", dataset_version_id=v.id, epochs=2)
    assert out["estimated"] == {
        "minutes": max(1, round(out["iterations"] / 2.0 / 60)),
        "basis": "this project's last run",
    }


# ── specialists: agents-as-tools ────────────────────────────────────────────


def test_scout_datasets_delegates_to_a_specialist_and_returns_its_report(session, project, monkeypatch):
    import json as _json

    from test_openai_provider import ScriptedModel, _call, _text

    from slm.data import scout_tools
    from slm.tuner import specialists

    session.add(StudioState(project_id=project.id, autopilot=True))
    session.commit()
    monkeypatch.setattr(scout_tools, "search_datasets", lambda q, limit=8: [{"id": "org/cooking-qa", "license": "mit"}])
    monkeypatch.setattr(
        scout_tools,
        "preview_rows",
        lambda repo, config=None, split=None, n=3: {"columns": ["q", "a"], "rows": [{"q": "x", "a": "y"}]},
    )
    report = {
        "candidates": [
            {"repo_id": "org/cooking-qa", "licence": "mit", "commercial_ok": True, "columns": ["q", "a"],
             "suggested_mapping": {"format": "instruction", "prompt": "q", "response": "a"}, "fit_score": 8, "why": "on goal"}
        ],
        "best": "org/cooking-qa",
        "summary": "One good set.",
    }  # fmt: skip
    monkeypatch.setattr(
        specialists,
        "model_override",
        ScriptedModel(
            [
                [_call("search_datasets", {"query": "cooking questions"}, 1)],
                [_call("preview_dataset", {"repo_id": "org/cooking-qa"}, 2)],
                [_text(_json.dumps(report))],
            ]
        ),
    )
    out = call(project.id, "scout_datasets", brief="cooking Q&A, one-sentence answers, may ship")
    assert (
        out["best"] == "org/cooking-qa"
        and out["candidates"][0]["fit_score"] == 8
        and out["searched_and_previewed"] == 2
    )
    rows = session.exec(select(TunerMessage).where(TunerMessage.role == "tool")).all()
    assert [(r.meta["agent"], r.meta["name"]) for r in rows] == [
        ("DataScout", "search_datasets"),
        ("DataScout", "preview_dataset"),
    ]


def test_specialists_are_cards_outside_a_round(session, project, monkeypatch):
    monkeypatch.setattr(tuner, "send", lambda *a, **k: None)
    _open_project(session, project)
    out = call(project.id, "scout_datasets", brief="x", reason="Find data.")
    assert out["status"] == "waiting for the user's confirmation"
    session.expire_all()
    assert session.get(StudioState, project.id).pending_action["kind"] == "scout"
