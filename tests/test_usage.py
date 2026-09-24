"""The OpenAI usage meter: self-metered from token counts, priced per model."""

from slm import usage
from slm.config import get_settings
from slm.db import ApiUsage


def test_cost_uses_the_price_table_and_cached_tokens(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "openai_price_input", 2.0)
    monkeypatch.setattr(s, "openai_price_cached", 0.5)
    monkeypatch.setattr(s, "openai_price_output", 8.0)
    # 1M fresh input = $2, 1M cached = $0.5, 1M output = $8
    assert usage.cost_usd(1_000_000, 0) == 2.0
    assert usage.cost_usd(1_000_000, 0, cached_tokens=1_000_000) == 0.5
    assert usage.cost_usd(0, 500_000) == 4.0
    assert usage.prices()["configured"] is True


def test_record_attributes_calls_to_the_scoped_project_and_summarises(session, project, client, monkeypatch):
    monkeypatch.setattr(get_settings(), "openai_api_key", "sk-proj-abcdefghijklmnop1234")
    with usage.scope(project.id, "evaluate"):
        usage.record("gpt-6-luna", 1000, 200)
        usage.record("gpt-6-luna", 500, 100, cached_tokens=400)
    usage.record("gpt-6-luna", 10_000, 2_000, requests=5, project_id=project.id, purpose="tuner")
    usage.record("gpt-6-luna", 100, 10)  # outside any scope: no project
    rows = session.query(ApiUsage).all()
    assert [(r.project_id, r.purpose, r.requests) for r in rows] == [
        (project.id, "evaluate", 1),
        (project.id, "evaluate", 1),
        (project.id, "tuner", 5),
        (None, "other", 1),
    ]
    out = client.get(f"/api/usage?project_id={project.id}").json()
    assert out["key"] == "sk-proj…1234" and out["model"]
    assert out["project"]["requests"] == 7 and out["project"]["input_tokens"] == 11_500
    assert out["project"]["cached_input_tokens"] == 400
    assert set(out["project"]["by_purpose"]) == {"evaluate", "tuner"}
    assert out["all_time"]["requests"] == 8 and out["all_time"]["cost_usd"] > out["project"]["cost_usd"]
    assert client.get("/api/usage").json()["all_time"]["input_tokens"] == 11_600


def test_the_meter_moves_on_the_jobs_feed(project):
    import asyncio

    from slm.events import bus

    got = []

    async def listen():
        async with bus.subscribe("jobs") as q:
            usage.record("gpt-6-luna", 1, 1, project_id=project.id)
            got.append(await asyncio.wait_for(q.get(), 1))

    asyncio.run(listen())
    assert got == [{"type": "usage", "project_id": project.id}]
