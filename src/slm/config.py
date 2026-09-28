"""Runtime settings, read from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Run from a clone (src/slm inside a checkout), the app keeps its workspace and .env in the repo, as
# it always has. Installed from PyPI, PROJECT_ROOT is somewhere inside Python's site-packages, so
# the app keeps them in the user's Application Support folder instead.
SOURCE_CHECKOUT = (PROJECT_ROOT / "pyproject.toml").is_file() and (PROJECT_ROOT / "src" / "slm").is_dir()
APP_HOME = PROJECT_ROOT if SOURCE_CHECKOUT else Path.home() / "Library" / "Application Support" / "SLM Forge"


# How to give SLM Forge a Hugging Face login, in every message that needs one.
HF_LOGIN_HELP = "put HF_TOKEN=hf_... in .env (a token that can write), or run `uv run hf auth login`"


def env_files() -> tuple[Path, ...]:
    """Where .env is read from, lowest priority first: the app home, then the current directory.
    SLM_DOTENV names one file instead (tests point it at nothing)."""
    import os

    if override := os.environ.get("SLM_DOTENV"):
        return (Path(override),)
    return tuple(dict.fromkeys([APP_HOME / ".env", Path.cwd() / ".env"]))


def env_file() -> Path:
    """The one .env file to tell people about: the app home's (the repo's .env in a clone,
    ~/Library/Application Support/SLM Forge/.env when installed), or SLM_DOTENV's."""
    import os

    return Path(override) if (override := os.environ.get("SLM_DOTENV")) else APP_HOME / ".env"


def key_help(key: str) -> str:
    """Where to put a missing key, naming the actual file (a bare ".env" left installed users guessing)."""
    return f"Add {key}=... to {env_file()} and restart SLM Forge"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=env_files(), env_prefix="SLM_", extra="ignore")

    workspace: Path = APP_HOME / "workspace"

    # Agent LLM provider. OpenAI is the default; Claude and a local Ollama model also work.
    agent_provider: Literal["openai", "claude", "ollama"] = "openai"
    openai_model: str = "gpt-6-luna"  # all agentic work: the Tuner and the helper agents
    # Agent traces (full prompts, tool calls and outputs) go to the OpenAI dashboard only if you opt in.
    openai_tracing: bool = False
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
    """An unprefixed key (OPENAI_API_KEY…) from the highest-priority .env that sets it."""
    for env_file in reversed(env_files()):
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                if value := line.split("=", 1)[1].strip().strip('"').strip("'"):
                    return value
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
    # Hugging Face reads HF_TOKEN from the environment; let it live in .env with the other keys.
    if not os.environ.get("HF_TOKEN") and (token := _dotenv_value("HF_TOKEN")):
        os.environ["HF_TOKEN"] = token
    return s


def agent_key_configured(s: Settings) -> bool:
    """The provider switch covers the AI judge and the Advanced screens' agents (Observer, Synth)."""
    return {"openai": bool(s.openai_api_key), "claude": bool(s.anthropic_api_key), "ollama": True}[s.agent_provider]


def tuner_ready(s: Settings) -> bool:
    """The Tuner and its specialists run on the OpenAI Agents SDK whatever SLM_AGENT_PROVIDER says."""
    return bool(s.openai_api_key)
