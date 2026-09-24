"""Runtime settings, read from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_prefix="SLM_", extra="ignore")

    workspace: Path = PROJECT_ROOT / "workspace"

    # Agent LLM provider. OpenAI is the default; Claude and a local Ollama model also work.
    agent_provider: Literal["openai", "claude", "ollama"] = "openai"
    openai_model: str = "gpt-6-luna"  # all agentic work: the Tuner and the helper agents
    openai_tracing: bool = True  # send agent traces to the OpenAI dashboard
    # The spend meter reads OpenAI's Costs API, which needs an organisation admin key; the project
    # id narrows it to the project the API key belongs to. Both also accepted unprefixed in .env.
    openai_admin_key: str | None = None
    openai_project_id: str | None = None
    claude_model: str = "claude-opus-5"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_url: str = "http://localhost:11434"

    # Also read under their standard unprefixed names (see get_settings).
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Above this many GB used by the app, the Disk pill suggests clearing intermediate run files.
    disk_tidy_gb: float = 20.0

    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def db_path(self) -> Path:
        return self.workspace / "slm.db"

    @property
    def datasets_dir(self) -> Path:
        return self.workspace / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.workspace / "runs"

    @property
    def exports_dir(self) -> Path:
        return self.workspace / "exports"

    @property
    def uploads_dir(self) -> Path:
        return self.workspace / "uploads"

    def ensure_dirs(self) -> None:
        for d in (
            self.workspace,
            self.datasets_dir,
            self.runs_dir,
            self.exports_dir,
            self.uploads_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def _dotenv_value(name: str) -> str | None:
    import os

    env_file = Path(os.environ.get("SLM_DOTENV") or PROJECT_ROOT / ".env")  # tests point it at nothing
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        if line.strip().startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


@lru_cache
def get_settings() -> Settings:
    import os

    s = Settings()
    # pydantic-settings only maps SLM_-prefixed names; accept the standard key names too.
    for field, name in (
        ("openai_api_key", "OPENAI_API_KEY"),
        ("anthropic_api_key", "ANTHROPIC_API_KEY"),
        ("openai_admin_key", "OPENAI_ADMIN_KEY"),
        ("openai_project_id", "OPENAI_PROJECT_ID"),
    ):
        if getattr(s, field) is None:
            setattr(s, field, os.environ.get(name) or _dotenv_value(name))
    return s


def agent_key_configured(s: Settings) -> bool:
    """The provider switch covers the AI judge and the Advanced screens' agents (Observer, Synth)."""
    return {"openai": bool(s.openai_api_key), "claude": bool(s.anthropic_api_key), "ollama": True}[s.agent_provider]


def tuner_ready(s: Settings) -> bool:
    """The Tuner and its specialists run on the OpenAI Agents SDK whatever SLM_AGENT_PROVIDER says."""
    return bool(s.openai_api_key)
