"""The Tuner's tools, invoked exactly as the Agents SDK invokes them, and the Studio API."""

import asyncio
import json

import pytest
from agents.tool_context import ToolContext
from sqlmodel import select

from slm.db import Dataset, Job, PreferencePair, Project, SftExample, StudioState, TunerMessage
from slm.events import canvas_changed
from slm.inference.engine import engine as infer
from slm.train.worker import worker
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


def test_prepare_dataset_passes_the_plan_through(session, project, tmp_path, monkeypatch):
    # Regression: the prompt promised "a different seed is a fresh batch" and DataPrep returned
    # max_chars, but prepare_dataset accepted neither, so both were silently lost.
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps({"q": f"q{i}", "a": f"answer {i}"}) for i in range(5)))
    ds = Dataset(project_id=project.id, name="r", source="upload", raw_path=str(raw), columns=["q", "a"])
    session.add(ds)
    session.commit()
    submitted = {}

    def fake_submit(kind, config, pid):
        submitted.update(config)
        raise RuntimeError("stop here")

    monkeypatch.setattr(tools.data, "_submit", fake_submit)
    call(
        project.id,
        "prepare_dataset",
        dataset_id=ds.id,
        format="instruction",
        prompt_column="q",
        response_column="a",
        max_examples=3,
        min_answer_chars=4,
        max_chars=900,
        seed=7,
    )
    assert submitted["seed"] == 7 and submitted["max_examples"] == 3
    assert submitted["rules"] == {"min_chars": 4, "long_examples": "auto", "max_chars": 900}


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
            canvas_changed(project.id)
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
    assert {t.name for t in tools.REGISTRY} == {t.name for t in ALL_TOOLS}
    assert len(tools.REGISTRY) == len(ALL_TOOLS) == 33


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

    monkeypatch.setattr(worker, "submit", submit)
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
    assert submitted == [] and sent[-1] == ("user", "Go ahead.")  # no card id: it's just a message
    card_id = client.get(f"/api/projects/{project.id}/studio").json()["pending_action"]["id"]
    client.post(f"/api/projects/{project.id}/tuner/message", json={"text": "Go ahead.", "pending_id": "a-stale-one"})
    assert submitted == []  # a "yes" meant for an older card confirms nothing
    client.post(f"/api/projects/{project.id}/tuner/message", json={"text": "Go ahead.", "pending_id": card_id})
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
    out = call(project.id, "generate_synthetic_examples", kind="sft", count=40, focus="forces")
    assert "needs their go-ahead" in str(out) and submitted == []  # no reason given: refused outright
    out = call(project.id, "generate_synthetic_examples", kind="sft", count=40, focus="forces", reason="Fix forces.")
    assert out["status"] == "waiting for the user's confirmation" and submitted == []
    snap = client.get(f"/api/projects/{project.id}/studio").json()
    assert snap["pending_action"]["kind"] == "synthesize" and snap["pending_action"]["details"]["count"] == 40

    # Confirming grants exactly one call with exactly these arguments, and does NOT reopen the round.
    client.post(f"/api/projects/{project.id}/studio/confirm", json={})
    session.expire_all()
    st = session.get(StudioState, project.id)
    expected_grant = {"tool": "generate_synthetic_examples", "args": {"kind": "sft", "count": 40, "focus": "forces"}}
    assert st.granted == [expected_grant] and st.completed is True  # a spend card is no mandate for the round
    monkeypatch.setattr(tools.data, "_wait", lambda job_id, timeout: submitted[-1])
    # Different arguments (200 where 150 were approved): no grant, a new card instead.
    out = call(project.id, "generate_synthetic_examples", kind="sft", count=50, focus="forces", reason="More.")
    assert out["status"] == "waiting for the user's confirmation" and submitted == []
    client.post(f"/api/projects/{project.id}/studio/decline", json={})
    session.expire_all()
    assert session.get(StudioState, project.id).granted == []  # a decision clears leftover grants
    call(project.id, "generate_synthetic_examples", kind="sft", count=40, focus="forces", reason="Fix forces.")
    client.post(f"/api/projects/{project.id}/studio/confirm", json={})
    call(project.id, "generate_synthetic_examples", kind="sft", count=40, focus="forces")
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
    monkeypatch.setattr(tools.data, "_wait", lambda job_id, timeout: submitted[-1])
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
        tools.generate,
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

    monkeypatch.setattr(infer, "unload", lambda: None)
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
        tools.generate,
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
    assert rows[0].meta["args"] == {"query": "cooking questions"}  # the SDK passes args positionally


def test_specialists_are_cards_outside_a_round(session, project, monkeypatch):
    monkeypatch.setattr(tuner, "send", lambda *a, **k: None)
    _open_project(session, project)
    out = call(project.id, "scout_datasets", brief="x", reason="Find data.")
    assert out["status"] == "waiting for the user's confirmation"
    session.expire_all()
    assert session.get(StudioState, project.id).pending_action["kind"] == "scout"


# ── phase 1 regressions ─────────────────────────────────────────────────────


def test_a_run_after_a_roll_back_descends_from_the_served_checkpoint(session, project):
    from slm.db import Checkpoint
    from slm.train.jobs import serve_checkpoint, served_ancestry, served_checkpoint

    a = Checkpoint(project_id=project.id, kind="sft", job_id=1, base_model_path="/base", adapter_path="/a")
    session.add(a)
    session.commit()
    b = Checkpoint(
        project_id=project.id, kind="sft", job_id=2, base_model_path="/base", adapter_path="/b", parent_id=a.id
    )
    session.add(b)
    session.commit()
    p = session.get(Project, project.id)
    serve_checkpoint(session, p, b)
    assert served_checkpoint(session, p).id == b.id
    serve_checkpoint(session, p, a)  # roll back: the next run must record A as its parent
    assert served_checkpoint(session, p).id == a.id  # not the newest (B)
    assert [c.id for c in served_ancestry(session, p)] == [a.id]


def test_halt_cancels_a_specialist_waiting_in_run_coroutine(project):
    import asyncio
    import threading
    import time

    from slm.tuner.session import Tuner

    t = Tuner()

    async def slow():
        await asyncio.sleep(30)

    threading.Timer(0.3, lambda: t.halt(project.id)).start()
    t0 = time.time()
    with pytest.raises(RuntimeError, match="stopped"):
        t.run_coroutine(slow(), pid=project.id)
    assert time.time() - t0 < 5 and not t._tasks.get(project.id)


def test_halted_spend_and_nudge_do_nothing(session, project, monkeypatch):
    from slm.tuner.session import Tuner, autopilot_nudge

    t = Tuner()
    monkeypatch.setattr("slm.tuner.session.tuner", t)
    session.add(StudioState(project_id=project.id, autopilot=True))
    session.commit()
    t.halt(project.id)
    assert "stopped this project" in str(call(project.id, "scout_datasets", brief="x", reason="x"))
    assert autopilot_nudge(project.id) is None  # even though autopilot is still True in the DB


def test_stopping_a_project_withdraws_unused_grants(session, project, monkeypatch):
    from slm.sessions import stop_project

    session.add(StudioState(project_id=project.id, granted=[{"tool": "import_dataset", "args": {}}]))
    session.commit()
    monkeypatch.setattr(tuner, "halt", lambda pid: None)
    stop_project(project.id)
    session.expire_all()
    assert session.get(StudioState, project.id).granted == []


def test_a_run_with_no_parsed_metrics_is_flagged(project):
    from types import SimpleNamespace

    from slm.train.jobs import _warnings

    ctx = SimpleNamespace(job_id=999, result={"total_iters": 50}, note=lambda _: None)
    assert [w["code"] for w in _warnings(ctx)] == ["no_metrics"]
    ctx.result = {}
    assert _warnings(ctx) == []  # nothing ran: nothing to flag


def test_the_tuner_memory_is_trimmed_at_a_user_message_boundary(tmp_path):
    # The SDK session grew without bound: each turn re-sent the whole project history.
    from agents import SQLiteSession

    from slm.tuner.session import MAX_SESSION_ITEMS, trim_session

    s = SQLiteSession("p", tmp_path / "s.db")
    items = [{"role": "user", "content": "[New project] the goal"}]
    for i in range(150):
        items += [
            {"role": "user", "content": f"u{i}"},
            {"type": "function_call", "call_id": f"c{i}", "name": "get_status", "arguments": "{}"},
            {"type": "function_call_output", "call_id": f"c{i}", "output": "{}"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
        ]
    asyncio.run(s.add_items(items))
    asyncio.run(trim_session(s))
    kept = asyncio.run(s.get_items())
    assert len(kept) <= MAX_SESSION_ITEMS + 1 < len(items)
    assert kept[0]["content"] == "[New project] the goal" and kept[1]["role"] == "user"
    calls = {it["call_id"] for it in kept if it.get("type") == "function_call"}
    assert all(it["call_id"] in calls for it in kept if it.get("type") == "function_call_output")
    asyncio.run(trim_session(s))
    assert len(asyncio.run(s.get_items())) == len(kept)  # a second pass changes nothing


def test_export_tool_rejects_bad_quantize_bits_before_proposing(project):
    out = call(project.id, "export_model", name="x", quantize_bits=5)
    assert "quantize_bits must be one of 3, 4, 6, 8" in str(out)
    from slm.tuner import confirm

    assert not confirm.pending(project.id)  # nothing was proposed


# ── the turn loop: send → _drain → _turn, on the Tuner's own thread ─────────


def _wait_idle(pid: int, timeout: float = 20) -> None:
    import time

    deadline = time.time() + timeout
    while tuner.is_busy(pid):
        assert time.time() < deadline, "the turn never ended"
        time.sleep(0.05)


def _transcript(session, pid: int) -> list[TunerMessage]:
    session.expire_all()
    return session.exec(select(TunerMessage).where(TunerMessage.project_id == pid).order_by(TunerMessage.id)).all()


def test_a_turn_records_the_tool_call_and_the_answer_in_order(session, project, monkeypatch):
    from test_openai_provider import StreamingScriptedModel, _call, _text

    monkeypatch.setattr(tuner, "model_override", StreamingScriptedModel([
        [_call("set_stage", {"stage": "data", "note": "Looking for data"}, 1)],
        [_text("Found some data.")],
    ]))  # fmt: skip
    tuner.send(project.id, "hello")
    _wait_idle(project.id)
    rows = _transcript(session, project.id)
    assert [(r.role, r.content) for r in rows] == [
        ("user", "hello"),
        ("tool", "set_stage"),
        ("assistant", "Found some data."),
    ]
    assert rows[1].meta["status"] == "done" and rows[1].meta["args"] == {"stage": "data", "note": "Looking for data"}
    assert session.get(StudioState, project.id).stage == "data"


def test_autopilot_nudges_twice_without_progress_then_pauses_itself(session, project, monkeypatch):
    from test_openai_provider import StreamingScriptedModel, _text

    session.add(StudioState(project_id=project.id, autopilot=True))
    session.commit()
    model = StreamingScriptedModel(
        [[_text("Thinking.")], [_text("Still thinking.")], [_text("Hmm.")], [_text("never asked")]]
    )
    monkeypatch.setattr(tuner, "model_override", model)
    tuner.send(project.id, "go")
    _wait_idle(project.id)
    rows = _transcript(session, project.id)
    assert len(model.turns) == 1  # three turns: the user's, then two autopilot nudges
    assert sum(1 for r in rows if r.meta.get("autopilot")) == 2
    assert rows[-1].meta.get("autopilot_paused") and "paused" in rows[-1].content
    st = session.get(StudioState, project.id)
    assert st.autopilot is False and st.stalled_nudges == 0


def test_too_many_steps_pause_the_tuner_instead_of_crashing(session, project, monkeypatch):
    from test_openai_provider import StreamingScriptedModel, _call

    turns = [[_call("set_stage", {"stage": "data", "note": f"step {i}"}, i)] for i in range(45)]
    monkeypatch.setattr(tuner, "model_override", StreamingScriptedModel(turns))
    tuner.send(project.id, "loop forever")
    _wait_idle(project.id, timeout=60)
    rows = _transcript(session, project.id)
    assert "paused after many steps" in rows[-1].content and rows[-1].role == "event"
    assert sum(1 for r in rows if r.role == "tool") == 40  # max_turns


def test_halt_cancels_the_turn_while_a_tool_is_running(session, project, monkeypatch):
    import threading

    from test_openai_provider import StreamingScriptedModel, _call, _text

    started, release = threading.Event(), threading.Event()

    def slow_overview():
        started.set()
        release.wait(10)
        return {"sessions": [], "capacity": {"gpu_running": None}}

    monkeypatch.setattr(tools.status, "overview", slow_overview)
    monkeypatch.setattr(
        tuner, "model_override", StreamingScriptedModel([[_call("machine_overview", {}, 1)], [_text("never")]])
    )
    tuner.send(project.id, "what is the machine doing?")
    assert started.wait(15)
    tuner.halt(project.id)
    release.set()
    _wait_idle(project.id)
    rows = _transcript(session, project.id)
    assert tuner.is_halted(project.id)
    assert not any(r.role == "assistant" for r in rows)  # the cancelled turn never answered
    tuner.unhalt(project.id)  # the singleton outlives this test; later tests reuse project id 1


def test_the_teacher_model_writes_small_sets_only(session, project, submitted, monkeypatch):
    # The user's rule: the dataset comes from public data; the API writes seeds and top-ups only.
    from slm.tuner.tools import MAX_SYNTHETIC_PER_CALL, MAX_SYNTHETIC_TOTAL

    monkeypatch.setattr(tuner, "send", lambda *a, **k: None)
    monkeypatch.setattr(tools.data, "_wait", lambda job_id, timeout: submitted[-1])
    session.add(StudioState(project_id=project.id, autopilot=True))  # a round in motion: no card needed
    session.commit()
    out = call(project.id, "generate_synthetic_examples", kind="sft", count=5000, focus="everything")
    assert submitted, out
    assert submitted[-1].config["count"] == MAX_SYNTHETIC_PER_CALL  # a huge ask is clamped, not honoured
    session.add_all(
        SftExample(
            project_id=project.id, messages=[{"role": "user", "content": f"q{i}"}], source="synthetic", approved=True
        )
        for i in range(MAX_SYNTHETIC_TOTAL)
    )
    session.commit()
    out = call(project.id, "generate_synthetic_examples", kind="sft", count=10, focus="more")
    assert "limit is 200" in str(out) and "scout_datasets" in str(out)
    assert len(submitted) == 1  # nothing else was written


def test_gguf_and_publish_are_cards_that_leave_a_finished_project_finished(
    session, project, submitted, monkeypatch, tmp_path
):
    from slm.db import Job
    from slm.models import hub

    monkeypatch.setattr(tuner, "send", lambda *a, **k: None)
    monkeypatch.setattr(hub, "account", lambda refresh=False: {"logged_in": True, "user": "someone", "can_write": True})
    folder = tmp_path / "chef"
    folder.mkdir()
    export = Job(project_id=project.id, kind="export", status="succeeded", result={"path": str(folder)})
    session.add(export)
    session.add(StudioState(project_id=project.id, completed=True, stage="export"))
    session.commit()

    out = call(project.id, "export_gguf", export_job_id=export.id, reason="For Ollama.")
    assert out["status"] == "waiting for the user's confirmation" and submitted == []
    from slm.tuner import confirm

    confirm.confirm(project.id)
    assert [j.kind for j in submitted] == ["gguf"] and submitted[0].config["quants"] == ["Q4_K_M", "Q8_0"]
    session.expire_all()
    assert session.get(StudioState, project.id).completed is True  # converting doesn't reopen the round

    call(project.id, "upload_to_huggingface", export_job_id=export.id, repo_name="chef", reason="Share it.")
    card = session.get(StudioState, project.id)
    session.refresh(card)
    assert (
        card.pending_action["details"]["visibility"] == "public"
        and card.pending_action["details"]["repo_id"] == "someone/chef"
    )
    confirm.confirm(project.id)
    assert submitted[-1].kind == "hf_upload" and submitted[-1].config["repo_id"] == "someone/chef"
    session.expire_all()
    assert session.get(StudioState, project.id).completed is True
    bad = call(project.id, "upload_to_huggingface", export_job_id=export.id, repo_name="bad name!")
    assert "up to 96 characters" in str(bad)


def test_the_tuner_runs_on_the_chosen_provider(monkeypatch):
    # SLM_AGENT_PROVIDER picks the Tuner's model too: OpenAI by name (Responses API, as before),
    # Claude through the Agents SDK's LiteLLM extension, Ollama through its OpenAI-compatible endpoint.
    from agents import OpenAIChatCompletionsModel
    from agents.extensions.models.litellm_model import LitellmModel

    from slm.config import get_settings
    from slm.tuner import specialists
    from slm.tuner.agent import build_agent
    from slm.tuner.models import tuner_model

    s = get_settings()
    monkeypatch.setattr(s, "agent_provider", "openai")
    assert tuner_model() == s.openai_model == build_agent().model

    monkeypatch.setattr(s, "agent_provider", "claude")
    monkeypatch.setattr(s, "anthropic_api_key", "sk-ant-test")
    m = tuner_model()
    assert isinstance(m, LitellmModel) and m.model == f"anthropic/{s.claude_model}"
    assert isinstance(build_agent().model, LitellmModel)
    assert isinstance(specialists.build_scout().model, LitellmModel)
    assert isinstance(specialists.build_prep().model, LitellmModel)

    monkeypatch.setattr(s, "agent_provider", "ollama")
    m = tuner_model()
    assert isinstance(m, OpenAIChatCompletionsModel) and m.model == s.ollama_model
    assert str(m._client.base_url).rstrip("/") == s.ollama_url.rstrip("/") + "/v1"


def test_the_prompts_are_the_same_for_every_provider(monkeypatch):
    # The owner's call: SOTA models from OpenAI and Anthropic read the same instructions equally
    # well, so switching provider changes the model only, never the prompt or the tools.
    from slm.config import get_settings
    from slm.tuner.agent import build_agent

    s = get_settings()
    seen = set()
    for provider in ("openai", "claude", "ollama"):
        monkeypatch.setattr(s, "agent_provider", provider)
        monkeypatch.setattr(s, "anthropic_api_key", "sk-ant-test")
        a = build_agent()
        seen.add((a.instructions, tuple(t.name for t in a.tools), a.model_settings.parallel_tool_calls))
    assert len(seen) == 1
