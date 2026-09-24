"""The OpenAI provider driven by the real Agents SDK Runner, with a scripted model (no network)."""

import json
from types import SimpleNamespace

import pytest
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from sqlmodel import select

from slm.agents import scout
from slm.agents.provider import OpenAIProvider, ProviderError, set_provider
from slm.config import get_settings
from slm.db import AgentEvent, Proposal


def _call(name: str, args: dict, n: int) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        type="function_call", name=name, arguments=json.dumps(args), call_id=f"call_{n}", id=f"fc_{n}"
    )


def _text(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        type="message", id="msg", role="assistant", status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )  # fmt: skip


class ScriptedModel(Model):
    """Returns one scripted turn per call, recording what the SDK sent."""

    def __init__(self, turns: list[list]):
        self.turns = turns
        self.seen_tools: list[str] = []

    async def get_response(self, system_instructions, input, model_settings, tools, *args, **kwargs):
        self.seen_tools = [t.name for t in tools]
        return ModelResponse(output=self.turns.pop(0), usage=Usage(), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def openai_settings(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "openai_api_key", "sk-test")
    monkeypatch.setattr(s, "openai_tracing", False)  # never export traces from tests
    return s


@pytest.fixture
def hf(monkeypatch):
    monkeypatch.setattr(scout.scout_tools, "search_datasets", lambda q, limit=10: [{"id": "org/cooking-qa"}])
    monkeypatch.setattr(
        scout.scout_tools,
        "preview_rows",
        lambda repo, config=None, split=None, n=3: {"columns": ["question", "answer"], "rows": [{"question": "q"}]},
    )


def test_agents_sdk_runner_drives_scout_tools(session, project, openai_settings, hf):
    mapping = {"format": "instruction", "prompt": "question", "response": "answer"}
    model = ScriptedModel(
        [
            [_text("Searching the Hub first."), _call("search_datasets", {"query": "cooking"}, 1)],
            [_call("preview_dataset", {"repo_id": "org/cooking-qa"}, 2)],
            [_call("propose_import", {"repo_id": "org/cooking-qa", "mapping": mapping, "rationale": "fits"}, 3)],
            [_call("propose_import", {"repo_id": "org/invented", "mapping": mapping, "rationale": "x"}, 4)],
            [_text("Proposed org/cooking-qa.")],
        ]
    )
    set_provider(OpenAIProvider(model=model))
    result = scout.run(project.id)

    assert result.text == "Proposed org/cooking-qa."
    assert result.steps == 5
    assert set(model.seen_tools) == {
        "search_datasets",
        "dataset_card",
        "preview_dataset",
        "propose_import",
        "propose_manual_acquisition",
    }
    # Same tool guard as the other providers: the invented dataset is rejected back to the model.
    assert [c["is_error"] for c in result.tool_calls] == [False, False, False, True]
    assert [p.payload["repo_id"] for p in session.exec(select(Proposal)).all()] == ["org/cooking-qa"]
    messages = [e.content["text"] for e in session.exec(select(AgentEvent).where(AgentEvent.kind == "message")).all()]
    assert "Searching the Hub first." in messages  # text between tool calls is surfaced


def test_json_uses_strict_structured_output(openai_settings, monkeypatch):
    provider = OpenAIProvider(model="test-model")
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(output_text='{"assessment": "ok", "notes": []}')

    monkeypatch.setattr(provider.client.responses, "create", create)
    out = provider.json(
        "sys",
        "user",
        {
            "type": "object",
            "properties": {"assessment": {"type": "string"}, "notes": {"type": "array", "items": {"type": "string"}}},
        },
    )
    assert out == {"assessment": "ok", "notes": []}
    fmt = sent["text"]["format"]
    assert fmt["strict"] is True and fmt["schema"]["additionalProperties"] is False
    assert fmt["schema"]["required"] == ["assessment", "notes"]


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(get_settings(), "openai_api_key", None)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAIProvider()
