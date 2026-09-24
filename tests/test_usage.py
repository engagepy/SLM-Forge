"""The OpenAI usage meter: self-metered from token counts, priced per model."""

from sqlmodel import select

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
    rows = session.exec(select(ApiUsage).order_by(ApiUsage.id)).all()
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


def test_a_tuner_turn_is_metered_per_model_call_as_it_streams(session, project, monkeypatch):
    # Regression: usage was read from the run only when the turn ended, so a long turn showed
    # nothing for minutes and a turn cut short by a restart was never counted.
    import asyncio

    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseUsage,
    )
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

    from slm.tuner.session import tuner

    def response():
        msg = ResponseOutputMessage(
            type="message", id="msg", role="assistant", status="completed",
            content=[ResponseOutputText(type="output_text", text="Hello.", annotations=[])],
        )  # fmt: skip
        return Response(
            id="r1", created_at=0, model="gpt-6-luna", object="response", output=[msg],
            parallel_tool_calls=False, tool_choice="auto", tools=[],
            usage=ResponseUsage(
                input_tokens=3000, output_tokens=50, total_tokens=3050,
                input_tokens_details=InputTokensDetails(cached_tokens=2000, cache_write_tokens=0),
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            ),
        )  # fmt: skip

    class StreamModel(Model):
        async def get_response(self, *a, **k):
            return ModelResponse(output=response().output, usage=Usage(), response_id=None)

        async def stream_response(self, *a, **k):
            yield ResponseCompletedEvent(type="response.completed", response=response(), sequence_number=0)

    from agents import set_tracing_disabled

    set_tracing_disabled(True)  # no key in tests; don't try to export a trace
    monkeypatch.setattr(tuner, "model_override", StreamModel())
    asyncio.run(tuner._turn(project.id, "hi"))
    rows = session.exec(select(ApiUsage)).all()
    assert [(r.project_id, r.purpose, r.input_tokens, r.cached_input_tokens, r.output_tokens) for r in rows] == [
        (project.id, "tuner", 3000, 2000, 50)
    ]
