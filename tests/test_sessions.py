from slm.db import Job, Metric, Project, StudioState
from slm.sessions import overview


def _proj(session, name, **state):
    p = Project(name=name, goal=name)
    session.add(p)
    session.commit()
    session.refresh(p)
    session.add(StudioState(project_id=p.id, **state))
    session.commit()
    return p


def test_overview_reports_each_state_progress_and_queue(session):
    running = _proj(session, "Physics")
    queued = _proj(session, "Cooking")
    _proj(session, "Support", autopilot=False)
    _proj(session, "Poems", completed=True)
    job = Job(project_id=running.id, kind="sft", status="running", result={"total_iters": 100})
    session.add(job)
    session.add(Job(project_id=queued.id, kind="sft", status="queued"))
    session.commit()
    session.add(Metric(job_id=job.id, iteration=40, split="train", values={"loss": 1.0, "it_per_sec": 0.5}))
    session.commit()

    data = overview()
    by_name = {s["name"]: s for s in data["sessions"]}
    assert by_name["Physics"]["state"] == "running"
    assert by_name["Physics"]["job"]["progress"] == {"current": 40, "total": 100, "percent": 40, "minutes_left": 2}
    assert by_name["Cooking"]["state"] == "queued" and by_name["Cooking"]["job"]["queue_position"] == 1
    assert by_name["Support"]["state"] == "paused"
    assert by_name["Poems"]["state"] == "done"
    cap = data["capacity"]
    assert cap["gpu_slots"] == 1 and cap["gpu_running"]["project"] == "Physics"
    assert [j["project"] for j in cap["gpu_queue"]] == ["Cooking"]
    assert "one model at a time" in cap["explanation"]


def test_stop_pauses_and_cancels_resume_reopens(client, session, monkeypatch):
    from slm.tuner.session import tuner

    woke = []
    monkeypatch.setattr(tuner, "send", lambda pid, *a, **k: woke.append(pid))
    monkeypatch.setattr(tuner, "is_busy", lambda pid: False)
    p = _proj(session, "Physics", completed=True)
    job = Job(project_id=p.id, kind="sft", status="queued")
    session.add(job)
    session.commit()
    c = client

    out = c.post(f"/api/sessions/{p.id}/stop", json={"cancel_jobs": True}).json()
    assert out == {"project_id": p.id, "autopilot": False, "cancelled_jobs": [job.id]}
    session.expire_all()
    assert session.get(Job, job.id).status == "cancelled"
    assert not session.get(StudioState, p.id).autopilot

    assert c.post(f"/api/sessions/{p.id}/resume").json()["autopilot"] is True
    session.expire_all()
    st = session.get(StudioState, p.id)
    assert st.autopilot and not st.completed and woke == [p.id]
    assert c.post("/api/sessions/999/stop", json={}).status_code == 404


def test_a_project_has_a_model_when_an_export_exists_on_disk(session, tmp_path):
    # Regression: a project exported from the Advanced screens never counted as having a model.
    p = _proj(session, "Chef")
    kept, gone = tmp_path / "chef-v1", tmp_path / "chef-v0"
    kept.mkdir()
    for path in (kept, gone):
        session.add(Job(project_id=p.id, kind="export", status="succeeded", result={"path": str(path)}))
    session.add(Job(project_id=p.id, kind="export", status="failed", result={"path": str(kept)}))
    session.commit()
    item = next(x for x in overview()["sessions"] if x["project_id"] == p.id)
    assert item["models"] == 1 and item["state"] != "done"  # a model, though the Tuner never finished
