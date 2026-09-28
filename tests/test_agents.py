import json

import pytest
from sqlmodel import select

from slm.agents import actions, observer, prep, scout, synth
from slm.agents.provider import FakeProvider, set_provider, strict_schema
from slm.db import (
    AgentEvent,
    Dataset,
    DatasetVersion,
    Feedback,
    Job,
    PreferencePair,
    Proposal,
    SftExample,
)


@pytest.fixture
def hf(monkeypatch):
    """Stub the Hub so agent tests run offline."""
    monkeypatch.setattr(
        scout.scout_tools,
        "search_datasets",
        lambda q, limit=10: [{"id": "org/cooking-qa", "license": "apache-2.0", "downloads": 10}],
    )
    monkeypatch.setattr(
        scout.scout_tools,
        "preview_rows",
        lambda repo, config=None, split=None, n=3: {
            "dataset": repo, "config": "default", "split": "train",
            "columns": ["question", "answer"], "rows": [{"question": "q", "answer": "a" * 700}],
        },
    )  # fmt: skip


def test_strict_schema_requires_all_and_closes_objects():
    s = strict_schema(
        {"type": "object", "properties": {"a": {"type": "object", "properties": {"b": {"type": "string"}}}}}
    )
    assert s["required"] == ["a"] and s["additionalProperties"] is False
    assert s["properties"]["a"]["required"] == ["b"]


def test_scout_proposes_only_datasets_it_has_seen(session, project, hf):
    mapping = {"format": "instruction", "prompt": "question", "response": "answer"}
    set_provider(
        FakeProvider(
            [
                {"tool": "search_datasets", "input": {"query": "cooking"}},
                {"tool": "preview_dataset", "input": {"repo_id": "org/cooking-qa"}},
                {
                    "tool": "propose_import",
                    "input": {"repo_id": "org/cooking-qa", "mapping": mapping, "rationale": "fits"},
                },
                {"tool": "propose_import", "input": {"repo_id": "org/made-up", "mapping": mapping, "rationale": "x"}},
                {"text": "Proposed one dataset."},
            ]
        )
    )
    result = scout.run(project.id)
    props = session.exec(select(Proposal)).all()
    assert [p.payload["repo_id"] for p in props] == ["org/cooking-qa"]
    assert props[0].status == "pending" and props[0].payload["mapping"] == mapping
    # The invented dataset is reported back to the model as a tool error, not silently accepted.
    assert [c["is_error"] for c in result.tool_calls] == [False, False, False, True]
    events = session.exec(select(AgentEvent).where(AgentEvent.kind == "tool_result")).all()
    preview_out = json.loads(events[1].content["output"])
    assert preview_out["rows"][0]["answer"].endswith("…")  # long values trimmed for context


def test_scout_manual_acquisition(session, project, hf):
    set_provider(
        FakeProvider(
            [
                {
                    "tool": "propose_manual_acquisition",
                    "input": {"title": "Get recipes", "instructions": "1. Download", "rationale": "gated"},
                }
            ]
        )
    )
    scout.run(project.id)
    prop = session.exec(select(Proposal)).one()
    assert prop.action == "acquire_manually"
    assert actions.execute(prop.id)["result"] == {"awaiting": "upload"}


def _pair(project_id, **kw):
    return PreferencePair(
        project_id=project_id,
        prompt=kw.get("prompt", "p"),
        chosen="good",
        rejected="bad",
        **{k: v for k, v in kw.items() if k != "prompt"},
    )


def test_observer_rules_propose_dpo_once(session, project):
    for i in range(observer.DPO_MIN_PAIRS):
        session.add(_pair(project.id, prompt=f"p{i}"))
    session.commit()
    out = observer.run(project.id, use_llm=False)
    assert out["rule_proposals"] == ["run_dpo"]
    assert observer.run(project.id, use_llm=False)["rule_proposals"] == []  # no duplicate while pending


def test_one_dpo_threshold_for_the_observer_and_the_ui():
    # Regression: the Observer proposed DPO at 8 pairs, the Refine page lit up at 8 and linked at 3,
    # and the Goal page said 30 (lesson 5). One number now, the same on both sides.
    import re
    from pathlib import Path

    api_ts = (Path(__file__).parents[1] / "web" / "src" / "api.ts").read_text()
    ui = int(re.search(r"export const DPO_MIN_PAIRS = (\d+);", api_ts).group(1))
    assert observer.DPO_MIN_PAIRS == ui == 30


def test_observer_llm_proposes_synthesis_and_marks_feedback(session, project):
    session.add(Feedback(project_id=project.id, prompt="how to boil pasta", candidate_a="a", candidate_b="b",
                         choice="both_bad", critique="too long and rambling"))  # fmt: skip
    session.commit()
    fake = FakeProvider(json_responses=[{
        "assessment": "Answers are too long.",
        "synthetic": {"should_generate": True, "kind": "sft", "count": 500, "focus": "short answers"},
        "notes": ["lower max_tokens"],
    }])  # fmt: skip
    set_provider(fake)
    out = observer.run(project.id)
    assert out["assessment"] == "Answers are too long." and out["observed"] == 1
    prop = session.exec(select(Proposal).where(Proposal.action == "generate_synthetic")).one()
    assert prop.payload["count"] == 60  # clamped
    assert "too long and rambling" in fake.json_calls[0][1]
    session.expire_all()
    assert session.exec(select(Feedback)).one().observed


def test_observer_surfaces_last_run_warnings(session, project):
    job = Job(
        project_id=project.id,
        kind="sft",
        status="succeeded",
        result={"warnings": [{"code": "diverged", "message": "Training loss spiked to 7.88 at iteration 25"}]},
    )
    session.add(job)
    session.commit()
    notes = observer.run(project.id, use_llm=False)["notes"]
    assert any("spiked" in n for n in notes)


def test_synth_blocks_contamination_and_saves_unapproved(session, project, tmp_path):
    vdir = tmp_path / "v1"
    vdir.mkdir()
    (vdir / "test.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "How long to boil an egg?"},
                    {"role": "assistant", "content": "9 min"},
                ]
            }
        )
        + "\n"
    )
    session.add(DatasetVersion(project_id=project.id, path=str(vdir)))
    session.commit()
    set_provider(FakeProvider(json_responses=[{"examples": [
        {"prompt": "how long to boil an EGG?", "ideal_response": "9 minutes", "weak_response": ""},  # eval prompt
        {"prompt": "How do I dice an onion?", "ideal_response": "Halve, slice, cross-cut.", "weak_response": ""},
        {"prompt": "How do I dice an onion?", "ideal_response": "dup", "weak_response": ""},
        {"prompt": "", "ideal_response": "x", "weak_response": ""},
    ]}]))  # fmt: skip
    stats = synth.run(project.id, kind="sft", count=4)
    assert stats["saved"] == 1 and stats["dropped_contamination"] == 1
    assert stats["dropped_duplicate"] == 1 and stats["dropped_empty"] == 1
    ex = session.exec(select(SftExample)).one()
    assert ex.approved is False and ex.source == "synthetic"


def test_synth_preference_falls_back_to_teacher_weak_answer(session, project):
    set_provider(FakeProvider(json_responses=[{"examples": [
        {"prompt": "Sear a steak?", "ideal_response": "Hot pan, dry steak, 2-3 min a side.", "weak_response": "Boil it."},
    ]}]))  # fmt: skip
    stats = synth.run(project.id, kind="preference", count=1, on_policy=False)
    pair = session.exec(select(PreferencePair)).one()
    assert stats["saved"] == 1 and pair.rejected == "Boil it." and pair.approved is False


def test_prep_uses_llm_mapping_when_it_validates(session, project, tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps({"q": f"question {i}", "a": f"answer {i}"}) for i in range(5)))
    ds = Dataset(project_id=project.id, name="qa", source="upload", raw_path=str(raw), columns=["q", "a"])
    session.add(ds)
    session.commit()
    set_provider(FakeProvider(json_responses=[{
        "format": "instruction", "prompt": "q", "response": "a", "messages": "", "input": "", "system": "",
        "text": "", "chosen": "", "rejected": "", "min_chars": 3, "max_chars": 5000, "rationale": "q/a pairs",
    }]))  # fmt: skip
    prop = prep.run(ds.id)
    assert prop.payload["mapping"] == {"format": "instruction", "prompt": "q", "response": "a"}
    assert prop.payload["rules"] == {"min_chars": 3, "max_chars": 5000}
    assert prop.payload["preview"][0]["messages"][1]["content"] == "answer 0"


def test_prep_falls_back_to_heuristic_when_llm_mapping_breaks(session, project, tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps({"question": "q", "answer": "a"}) for _ in range(4)))
    ds = Dataset(project_id=project.id, name="qa", source="upload", raw_path=str(raw), columns=["question", "answer"])
    session.add(ds)
    session.commit()
    set_provider(FakeProvider(json_responses=[{"format": "chat", "messages": "nonexistent", "rationale": "wrong"}]))
    prop = prep.run(ds.id)
    assert prop.payload["mapping"]["format"] == "instruction"


def test_approving_synthesis_submits_a_job_and_rejection_is_final(session, project):
    from slm.agents.base import create_proposal

    prop = create_proposal(project.id, "observer", "generate_synthetic", "gen", "why", {"kind": "sft", "count": 5})
    out = actions.execute(prop.id)
    job = session.get(Job, out["result"]["job_id"])
    assert job.kind == "synthesize" and job.config["proposal_id"] == prop.id
    with pytest.raises(ValueError):
        actions.execute(prop.id)  # already approved


def test_manual_prepare_closes_pending_dataprep_suggestion(session, project, tmp_path, monkeypatch):
    from slm.agents.base import create_proposal
    from slm.train import jobs
    from slm.train.worker import JobContext

    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps({"q": f"question {i}", "a": f"answer number {i}"}) for i in range(8)))
    ds = Dataset(project_id=project.id, name="qa", source="upload", raw_path=str(raw), columns=["q", "a"])
    session.add(ds)
    session.commit()
    prop = create_proposal(
        project.id, "prep", "prepare_dataset", "Prepare qa", "", {"dataset_id": ds.id, "mapping": {}}
    )
    other = create_proposal(
        project.id, "prep", "prepare_dataset", "Prepare other", "", {"dataset_id": 999, "mapping": {}}
    )

    job = Job(
        project_id=project.id,
        kind="prepare_dataset",
        config={"dataset_id": ds.id, "mapping": {"format": "instruction", "prompt": "q", "response": "a"}},
    )
    session.add(job)
    session.commit()
    ctx = JobContext(job, worker=None)
    jobs.prepare_dataset_job(ctx)

    session.expire_all()
    assert session.get(Proposal, prop.id).status == "executed"
    assert session.get(Proposal, prop.id).result["dataset_version_id"] == ctx.result["dataset_version_id"]
    assert session.get(Proposal, other.id).status == "pending"  # other datasets untouched
    assert ctx.result["n_train"] >= 1 and ctx.result["kept"] == 8
