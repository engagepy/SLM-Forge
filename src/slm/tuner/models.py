"""Which model the Tuner and its specialists (DataScout, DataPrep) run on.

`SLM_AGENT_PROVIDER` picks it for the whole app, the AI judge and the Advanced agents included:

- `openai` (default): `SLM_OPENAI_MODEL` on the Responses API, exactly as before.
- `claude`: `SLM_CLAUDE_MODEL` through the Agents SDK's LiteLLM extension.
- `ollama` (experimental): `SLM_OLLAMA_MODEL` through Ollama's OpenAI-compatible endpoint. It runs,
  but a small local model rarely drives a long, tool-heavy loop well.

The prompts are the same for all three. The SDK's Agent, Runner, sessions, streaming and
structured outputs don't change; only the model object handed to `Agent(model=...)` does.
"""

from agents import OpenAIChatCompletionsModel, set_tracing_disabled
from agents.models.interface import Model

from slm.config import Settings, get_settings, key_help

# The key each provider needs (None: no key), and the providers that are experimental for the Tuner.
TUNER_KEY: dict[str, str | None] = {"openai": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY", "ollama": None}
EXPERIMENTAL = {"ollama"}


def tuner_model_name(s: Settings | None = None) -> str:
    s = s or get_settings()
    return {"openai": s.openai_model, "claude": s.claude_model, "ollama": s.ollama_model}[s.agent_provider]


def tuner_model(s: Settings | None = None) -> str | Model:
    """What to pass as `Agent(model=...)`: a model name for OpenAI, a model object otherwise."""
    s = s or get_settings()
    if s.agent_provider == "claude":
        import litellm
        from agents.extensions.models.litellm_model import LitellmModel

        # Drop a setting the provider doesn't support (parallel_tool_calls on some models) instead of
        # failing the turn. Claude supports it: LiteLLM maps it to disable_parallel_tool_use.
        litellm.drop_params = True
        return LitellmModel(model=f"anthropic/{s.claude_model}", api_key=s.anthropic_api_key)
    if s.agent_provider == "ollama":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=f"{s.ollama_url.rstrip('/')}/v1", api_key="ollama")  # Ollama ignores the key
        return OpenAIChatCompletionsModel(model=s.ollama_model, openai_client=client)
    return s.openai_model


def prepare_sdk(s: Settings | None = None) -> None:
    """Before a Tuner turn: set the SDK up for the provider, or raise a clear error naming the
    missing key and the exact file it goes in."""
    from slm.agents.provider import OpenAIProvider, ProviderError

    s = s or get_settings()
    if s.agent_provider == "openai":
        OpenAIProvider()  # sets the SDK's key and tracing, or raises
        return
    if s.agent_provider == "claude" and not s.anthropic_api_key:
        raise ProviderError(f"No Anthropic API key. {key_help('ANTHROPIC_API_KEY')}.")
    # Traces only ever go to OpenAI's dashboard; with another provider there's nowhere to send them.
    set_tracing_disabled(True)
