"""The disk footprint: everything the app stores locally, by part."""

from slm import storage
from slm.config import get_settings
from slm.db import ModelRecord


def test_footprint_sums_workspace_parts_and_downloaded_models(session, tmp_path, monkeypatch, client):
    s = get_settings()
    s.ensure_dirs()
    (s.datasets_dir / "p1").mkdir(parents=True, exist_ok=True)
    (s.datasets_dir / "p1" / "raw.jsonl").write_bytes(b"x" * 3_000_000)
    (s.exports_dir / "chef").mkdir(parents=True, exist_ok=True)
    (s.exports_dir / "chef" / "weights.safetensors").write_bytes(b"y" * 5_000_000)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b"z" * 1_000_000)
    session.add(ModelRecord(repo_id="org/tiny", local_path=str(model)))
    session.commit()
    storage._cache.clear()
    fp = storage.footprint()
    assert fp["parts_gb"]["datasets"] == round(3_000_000 / storage.GB, 3)
    assert fp["parts_gb"]["exports"] == round(5_000_000 / storage.GB, 3)
    assert fp["parts_gb"]["models"] == round(1_000_000 / storage.GB, 3)
    assert fp["total_gb"] == round(9_000_000 / storage.GB, 3) and fp["disk_free_gb"] > 0
    # Cached for a minute, so the header's polling doesn't walk the disk every time.
    (s.exports_dir / "chef" / "more").write_bytes(b"w" * 5_000_000)
    assert storage.footprint()["parts_gb"]["exports"] == fp["parts_gb"]["exports"]
    assert storage.footprint(refresh=True)["parts_gb"]["exports"] > fp["parts_gb"]["exports"]
    assert client.get("/api/system").json()["disk"]["workspace"] == str(s.workspace)


def _project_with_files(session, name="Chef"):
    from slm.db import Job, Project

    s = get_settings()
    s.ensure_dirs()
    session.expunge_all()  # rows deleted through the API may get their ids reused
    p = Project(name=name, goal="g", base_model="org/tiny")
    session.add(p)
    session.commit()
    session.refresh(p)
    run = Job(project_id=p.id, kind="sft", status="succeeded")
    session.add(run)
    session.commit()
    session.refresh(run)
    (s.runs_dir / f"job-{run.id:05d}" / "adapters").mkdir(parents=True, exist_ok=True)  # the workspace outlives a test
    (s.runs_dir / f"job-{run.id:05d}" / "adapters" / "a.safetensors").write_bytes(b"r" * 2_000_000)
    (s.datasets_dir / f"p{p.id}").mkdir(parents=True, exist_ok=True)
    (s.datasets_dir / f"p{p.id}" / "raw.jsonl").write_bytes(b"d" * 1_000_000)
    export_dir = s.exports_dir / f"{name.lower()}-v1"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "weights").write_bytes(b"e" * 4_000_000)
    exp = Job(project_id=p.id, kind="export", status="succeeded", result={"path": str(export_dir)})
    session.add(exp)
    session.commit()
    session.refresh(exp)
    return p, run, exp, export_dir


def test_inventory_lists_models_with_their_users_and_projects_with_sizes(session, tmp_path, client):
    model = tmp_path / "m"
    model.mkdir()
    (model / "w").write_bytes(b"m" * 3_000_000)
    session.add(ModelRecord(repo_id="org/tiny", local_path=str(model), origin="downloaded"))
    session.commit()
    p, run, exp, _ = _project_with_files(session)
    inv = client.get("/api/storage").json()
    m = inv["models"][0]
    assert m["repo_id"] == "org/tiny" and m["used_by"] == [{"id": p.id, "name": "Chef"}] and m["origin"] == "downloaded"
    pr = inv["projects"][0]
    assert pr["parts_gb"]["runs"] == round(2_000_000 / storage.GB, 3) and pr["parts_gb"]["datasets"] == round(
        1_000_000 / storage.GB, 3
    )
    assert pr["exports"][0]["name"] == "chef-v1" and pr["exports"][0]["size_gb"] == round(4_000_000 / storage.GB, 3)


def test_deleting_an_export_frees_its_folder_and_hides_it(session, client):
    from slm.sessions import export_jobs

    p, run, exp, export_dir = _project_with_files(session)
    out = client.delete(f"/api/projects/{p.id}/exports/{exp.id}").json()
    assert out["freed_gb"] == round(4_000_000 / storage.GB, 3) and not export_dir.exists()
    session.expire_all()
    assert export_jobs(session, p.id) == []  # gone from Home, Try it and the Studio
    assert client.delete(f"/api/projects/{p.id}/exports/{run.id}").status_code == 404  # not an export


def test_deleting_a_project_removes_rows_files_and_memory_but_can_keep_exports(session, client, monkeypatch):
    from sqlmodel import select

    from slm.db import Job, Project
    from slm.tuner.session import tuner

    forgotten = []
    monkeypatch.setattr(tuner, "forget", lambda pid: forgotten.append(pid))
    monkeypatch.setattr(tuner, "halt", lambda pid: None)
    p, run, exp, export_dir = _project_with_files(session)
    pid, run_id = p.id, run.id  # the rows are gone after the delete; the instances can't be read then
    s = get_settings()
    out = client.delete(f"/api/projects/{pid}?keep_exports=true").json()
    assert out["freed_gb"] == round(3_000_000 / storage.GB, 3) and out["kept_exports"] is True
    assert export_dir.exists() and not (s.runs_dir / f"job-{run_id:05d}").exists()
    session.expire_all()
    assert session.exec(select(Project)).all() == [] and session.exec(select(Job)).all() == []
    assert forgotten == [pid]
    assert client.delete(f"/api/projects/{pid}").status_code == 404
    # Without keep_exports the exported model goes too.
    p2, _, _, export2 = _project_with_files(session, "Poet")
    pid2 = p2.id
    out = client.delete(f"/api/projects/{pid2}").json()
    assert not export2.exists() and out["freed_gb"] == round(7_000_000 / storage.GB, 3)


def test_removing_a_base_model_is_refused_while_a_project_uses_it(session, tmp_path, client):
    model = tmp_path / "m"
    model.mkdir()
    (model / "w").write_bytes(b"m" * 3_000_000)
    session.add(ModelRecord(repo_id="org/tiny", local_path=str(model)))
    session.commit()
    p, *_ = _project_with_files(session)
    r = client.delete("/api/models/org/tiny")
    assert r.status_code == 409 and "Chef" in r.json()["detail"]
    p.base_model = None
    session.add(p)
    session.commit()
    out = client.delete("/api/models/org/tiny").json()
    assert out["freed_gb"] == round(3_000_000 / storage.GB, 3) and not model.exists()
    session.expire_all()
    assert session.exec(__import__("sqlmodel").select(ModelRecord)).all() == []
    assert client.delete("/api/models/org/tiny").status_code == 404
