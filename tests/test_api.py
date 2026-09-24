import json

from sqlmodel import select

from slm.db import Job, PreferencePair, SftExample


def _project(client, **kw) -> int:
    r = client.post("/api/projects", json={"name": "cook", "goal": "cooking help"} | kw)
    assert r.status_code == 200
    return r.json()["id"]


def test_project_crud_and_overview(client):
    pid = _project(client)
    assert (
        client.patch(f"/api/projects/{pid}", json={"system_prompt": "Be brief."}).json()["system_prompt"] == "Be brief."
    )
    ov = client.get(f"/api/projects/{pid}").json()
    assert ov["counts"]["feedback"] == 0 and ov["base_model_downloaded"] is False
    assert client.get("/api/projects/999").status_code == 404


def test_upload_detects_mapping_and_previews(client):
    pid = _project(client)
    body = "\n".join(json.dumps({"instruction": f"q{i}", "output": f"a{i}"}) for i in range(6))
    r = client.post(f"/api/projects/{pid}/datasets/upload", files={"file": ("qa.jsonl", body)}, data={"name": "qa"})
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["n_rows"] == 6 and ds["suggested_mapping"]["format"] == "instruction"
    prev = client.post(
        f"/api/projects/{pid}/datasets/{ds['id']}/mapping-preview", json={"mapping": ds["suggested_mapping"]}
    ).json()
    assert prev["failed"] == 0 and prev["records"][0]["messages"][0]["content"] == "q0"


def test_upload_rejects_unknown_file_type(client):
    pid = _project(client)
    r = client.post(f"/api/projects/{pid}/datasets/upload", files={"file": ("x.exe", b"MZ")})
    assert r.status_code == 400


def _fb(client, pid, **kw):
    body = {"prompt": "How to boil pasta?", "candidate_a": "Salt water, 10 min.", "candidate_b": "Idk."} | kw
    return client.post(f"/api/projects/{pid}/feedback", json=body)


def test_feedback_turns_into_training_signal(client, session):
    pid = _project(client)
    assert _fb(client, pid, choice="a").json() == {"feedback_id": 1, "preference_pairs": 1, "sft_examples": 0}
    # Both bad + an edit: the edit beats both candidates and becomes an SFT example.
    r = _fb(client, pid, choice="both_bad", edited_answer="Boil salted water, cook 8-10 min, drain.").json()
    assert r["preference_pairs"] == 2 and r["sft_examples"] == 1
    assert _fb(client, pid, choice="tie").json()["preference_pairs"] == 0
    assert _fb(client, pid, choice="both_bad").status_code == 422  # needs an edit or critique
    pairs = session.exec(select(PreferencePair)).all()
    assert pairs[0].chosen == "Salt water, 10 min." and pairs[0].rejected == "Idk."
    assert {p.chosen for p in pairs[1:]} == {"Boil salted water, cook 8-10 min, drain."}


def test_review_approves_and_deletes(client, session):
    pid = _project(client)
    session.add_all(
        [
            PreferencePair(project_id=pid, prompt="p", chosen="c", rejected="r", source="synthetic", approved=False),
            SftExample(project_id=pid, messages=[{"role": "user", "content": "q"}], source="synthetic", approved=False),
        ]
    )
    session.commit()
    pending = client.get(f"/api/projects/{pid}/examples").json()
    assert len(pending["pairs"]) == 1 and len(pending["sft"]) == 1
    client.post(f"/api/projects/{pid}/examples/review", json={"pair_ids": [pending["pairs"][0]["id"]]})
    client.post(f"/api/projects/{pid}/examples/review", json={"sft_ids": [pending["sft"][0]["id"]], "approve": False})
    assert client.get(f"/api/projects/{pid}/examples").json() == {"pairs": [], "sft": []}
    assert len(client.get(f"/api/projects/{pid}/examples?status=ready").json()["pairs"]) == 1


def test_agent_endpoints_queue_jobs(client, session):
    pid = _project(client)
    job_id = client.post(f"/api/projects/{pid}/agents/scout", json={"request": "recipes"}).json()["job_id"]
    job = session.get(Job, job_id)
    assert job.kind == "agent_scout" and job.status == "queued" and job.config == {"request": "recipes"}
    assert client.post(f"/api/projects/{pid}/agents/synthesize", json={"count": 0}).status_code == 422
    assert client.post(f"/api/jobs/{job_id}/cancel").json() == {"status": "cancelled"}


def test_training_requires_downloaded_model(client):
    pid = _project(client, base_model="mlx-community/not-downloaded")
    r = client.post(f"/api/projects/{pid}/train/sft", json={"train": {"iters": 5}, "dataset_version_id": 1})
    assert r.status_code == 409


def test_openapi_lists_all_route_groups(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/system", "/api/projects/{project_id}/compare", "/api/proposals/{proposal_id}/approve",
              "/api/jobs/{job_id}/stream", "/api/projects/{project_id}/train/presets"):  # fmt: skip
        assert p in paths


def test_try_prompts_are_scoped_and_skip_used_ones(client, session, project):
    from slm.agents.provider import FakeProvider, set_provider

    set_provider(
        FakeProvider(
            json_responses=[
                {
                    "prompts": [
                        {"text": "How long do I boil an egg?", "kind": "on-goal", "expect": "a time in minutes"},
                        {"text": "How long do I boil an egg?", "kind": "on-goal", "expect": "dup"},
                        {
                            "text": "What's the weather today?",
                            "kind": "should-not",
                            "expect": "says it only does cooking",
                        },
                        {"text": "Rest a steak how long?", "kind": "other", "expect": "minutes"},
                        {"text": "Extra beyond four", "kind": "on-goal", "expect": "x"},
                        {"text": "Sixth", "kind": "on-goal", "expect": "x"},
                    ]
                }
            ]
        )
    )
    r = client.post(f"/api/projects/{project.id}/try/prompts", json={"used": ["how long do I boil an egg?"]}).json()
    texts = [p["text"] for p in r["prompts"]]
    assert "How long do I boil an egg?" not in texts  # already used (case-insensitive)
    assert len(r["prompts"]) <= 4 and r["prompts"][0]["kind"] == "should-not"
    assert r["prompts"][1]["kind"] == "on-goal"  # unknown kinds fall back to on-goal


def test_system_reports_the_tuner_needs_openai_whatever_the_provider(client, monkeypatch):
    # Regression: SLM_AGENT_PROVIDER=claude reported "configured" while every Tuner turn failed.
    from slm.config import get_settings

    monkeypatch.setattr(get_settings(), "agent_provider", "ollama")
    monkeypatch.setattr(get_settings(), "openai_api_key", None)
    agents = client.get("/api/system").json()["agents"]
    assert agents["key_configured"] is True  # ollama needs no key
    assert agents["tuner"] == {"ready": False, "key_env": "OPENAI_API_KEY", "model": get_settings().openai_model}


def test_rollback_is_refused_while_a_job_uses_the_model(client, session):
    # Regression: the Tuner's serve_checkpoint refused this, the Advanced route didn't.
    from slm.db import Checkpoint

    pid = _project(client)
    c = Checkpoint(project_id=pid, kind="sft", job_id=1, adapter_path="/a", base_model_path="/m")
    session.add(c)
    session.add(Job(id=77, project_id=pid, kind="sft", status="running"))
    session.commit()
    r = client.post(f"/api/projects/{pid}/checkpoints/{c.id}/activate")
    assert r.status_code == 409 and "Job 77" in r.json()["detail"]
    # ...and a checkpoint from another project can't be generated from through this one.
    other = _project(client)
    body = {"messages": [{"role": "user", "content": "hi"}], "target": f"checkpoint:{c.id}"}
    assert client.post(f"/api/projects/{other}/generate", json=body).status_code == 404


def test_cancelling_a_finished_job_says_so(client, session):
    pid = _project(client)
    session.add(Job(id=78, project_id=pid, kind="sft", status="succeeded"))
    session.commit()
    r = client.post("/api/jobs/78/cancel")
    assert r.status_code == 409 and "already succeeded" in r.json()["detail"]
    assert client.post("/api/jobs/9999/cancel").status_code == 404


def test_the_export_route_rejects_bad_quantize_bits(client, project):
    r = client.post(f"/api/projects/{project.id}/export", json={"quantize_bits": 5})
    assert r.status_code in (409, 422)  # 409 when no model is downloaded yet, 422 for the bits
