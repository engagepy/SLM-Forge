from fastapi.testclient import TestClient

from slm import profile
from slm.api.app import app


def test_profile_starts_unknown_and_prompt_asks_to_infer():
    p = profile.get()
    assert p.level == "unknown" and p.notes == []
    assert "set_user_level" in profile.prompt_section(p)


def test_level_and_notes_persist_and_shape_the_prompt():
    profile.set_level("expert", "asked about DPO beta and LoRA rank")
    profile.remember("prefers the smallest model that works", "preference", project_id=3)
    profile.remember("Prefers the smallest model that works", "preference")  # same note, different case
    section = profile.prompt_section()
    assert "Level: **expert** (asked about DPO beta and LoRA rank)" in section
    assert "LoRA rank/scale" in section  # expert register
    assert section.count("smallest model that works") == 1  # deduplicated
    assert "exported SLM" in section  # the objective holds at every level


def test_beginner_register_differs_from_expert():
    profile.set_level("beginner")
    assert "plain language" in profile.prompt_section()
    assert "LoRA rank/scale" not in profile.prompt_section()


def test_bad_values_rejected():
    import pytest

    with pytest.raises(ValueError):
        profile.set_level("guru")
    with pytest.raises(ValueError):
        profile.remember("x", "mood")


def test_profile_api_lets_the_user_correct_and_forget():
    c = TestClient(app)
    note = profile.remember("wants 4-bit exports", "preference")
    assert c.post("/api/profile/level", json={"level": "intermediate"}).json()["level_evidence"] == "set by you"
    assert [n["text"] for n in c.get("/api/profile").json()["notes"]] == ["wants 4-bit exports"]
    assert c.delete(f"/api/profile/notes/{note['id']}").json()["notes"] == []
    assert c.delete("/api/profile/notes/nope").status_code == 404
    assert c.post("/api/profile/reset").json()["level"] == "unknown"


def test_tuner_tools_write_the_profile(project):
    import asyncio
    import json

    from agents.tool_context import ToolContext

    from slm.tuner.tools import ALL_TOOLS, TunerContext

    tools = {t.name: t for t in ALL_TOOLS}

    def call(name, **args):
        ctx = ToolContext(
            context=TunerContext(project.id), tool_name=name, tool_call_id="t", tool_arguments=json.dumps(args)
        )
        return asyncio.run(tools[name].on_invoke_tool(ctx, json.dumps(args)))

    call("set_user_level", level="expert", evidence="talks about effective batch size")
    call("remember_about_user", text="teaches high-school physics", kind="fact")
    p = profile.get()
    assert p.level == "expert" and p.notes[-1]["project_id"] == project.id
