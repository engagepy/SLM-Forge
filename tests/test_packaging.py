"""Installed from PyPI, the app must find its UI, its .env and its workspace without a source tree.

Regression: every path hung off the source tree, so an installed `slm serve` looked for .env and
the workspace inside site-packages and served no UI; and on a non-Apple machine it failed deep in
an MLX import instead of saying why.
"""

import platform
import sys

import pytest

from slm import cli, config


def test_the_current_directory_env_wins_over_the_app_home(tmp_path, monkeypatch):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    (home / ".env").write_text("OPENAI_API_KEY=from-home\nANTHROPIC_API_KEY=only-home\n")
    (cwd / ".env").write_text("OPENAI_API_KEY=from-cwd\n")
    monkeypatch.setattr(config, "env_files", lambda: (home / ".env", cwd / ".env"))
    assert config._dotenv_value("OPENAI_API_KEY") == "from-cwd"
    assert config._dotenv_value("ANTHROPIC_API_KEY") == "only-home"
    assert config._dotenv_value("OPENAI_ADMIN_KEY") is None


def test_env_files_default_to_the_app_home_then_the_current_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("SLM_DOTENV", raising=False)
    monkeypatch.chdir(tmp_path)
    assert config.env_files() == (config.APP_HOME / ".env", tmp_path / ".env")
    monkeypatch.setenv("SLM_DOTENV", "/nowhere/.env")
    assert config.env_files() == (config.Path("/nowhere/.env"),)


def test_a_checkout_keeps_its_workspace_in_the_repo():
    # The owner's real projects live in ./workspace of the clone; packaging must not move them.
    assert config.SOURCE_CHECKOUT and config.APP_HOME == config.PROJECT_ROOT


@pytest.mark.parametrize("plat,machine", [("linux", "x86_64"), ("darwin", "x86_64"), ("win32", "AMD64")])
def test_the_cli_says_it_needs_apple_silicon(monkeypatch, plat, machine):
    monkeypatch.setattr(sys, "platform", plat)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(sys, "argv", ["slm", "hardware"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert "Apple Silicon" in str(e.value)


def test_the_packaged_ui_is_served_before_a_dev_build():
    from slm.api import app

    assert app.PACKAGED_UI.name == "web_dist" and app.PACKAGED_UI.parent.name == "slm"
    if (app.PACKAGED_UI / "index.html").exists():
        assert app.WEB_DIST == app.PACKAGED_UI


def test_an_hf_token_in_env_reaches_hugging_face(tmp_path, monkeypatch):
    # Keys live in one place: HF_TOKEN in .env is what huggingface_hub uses, unless the shell set one.
    from huggingface_hub import get_token

    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_env_file\n")
    monkeypatch.setattr(config, "env_files", lambda: (tmp_path / ".env",))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config.get_settings.cache_clear()
    try:
        config.get_settings()
        assert get_token() == "hf_from_env_file"
        monkeypatch.setenv("HF_TOKEN", "hf_from_shell")
        config.get_settings.cache_clear()
        config.get_settings()
        assert get_token() == "hf_from_shell"
    finally:
        config.get_settings.cache_clear()


def test_a_missing_key_names_the_exact_file(client, monkeypatch, tmp_path):
    # Regression: an installed copy said "set OPENAI_API_KEY in .env" and the user couldn't tell
    # which .env; the app now names the file it reads (the app home's, or SLM_DOTENV's).
    import pytest

    from slm.agents.provider import OpenAIProvider, ProviderError

    target = tmp_path / "home" / ".env"
    monkeypatch.setenv("SLM_DOTENV", str(target))
    assert config.env_file() == target
    assert client.get("/api/system").json()["agents"]["env_file"] == str(target)
    monkeypatch.delenv("SLM_DOTENV")
    assert config.env_file() == config.APP_HOME / ".env"
    monkeypatch.setattr(config.get_settings(), "openai_api_key", None)
    with pytest.raises(ProviderError, match="Add OPENAI_API_KEY=... to .*\\.env and restart"):
        OpenAIProvider()
