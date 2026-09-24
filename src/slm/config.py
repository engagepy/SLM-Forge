"""Runtime settings, read from environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_prefix="SLM_", extra="ignore")

    workspace: Path = PROJECT_ROOT / "workspace"

    # Agent LLM provider. Claude is the default; Ollama is the offline fallback.
    agent_provider: Literal["claude", "ollama"] = "claude"
    claude_model: str = "claude-opus-5"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_url: str = "http://localhost:11434"

    # Read without the SLM_ prefix so the standard variable name works.
    anthropic_api_key: str | None = None

    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def db_path(self) -> Path:
        return self.workspace / "slm.db"

    @property
    def models_dir(self) -> Path:
        return self.workspace / "models"

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
            self.models_dir,
            self.datasets_dir,
            self.runs_dir,
            self.exports_dir,
            self.uploads_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    import os

    s = Settings()
    if s.anthropic_api_key is None:
        s.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if s.anthropic_api_key is None:
        # pydantic-settings only maps SLM_-prefixed names; read the plain name from .env too.
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    s.anthropic_api_key = line.split("=", 1)[1].strip().strip('"')
    return s
