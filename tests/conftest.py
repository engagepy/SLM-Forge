import os
import tempfile

import pytest

# Isolate every test run from the real workspace, the real HF cache and the real keys before slm
# modules read settings: a test must never see a downloaded model, a key or the user's .env.
os.environ["SLM_WORKSPACE"] = tempfile.mkdtemp(prefix="slm-test-")
os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="slm-test-hf-")
os.environ["SLM_DOTENV"] = "/nonexistent/.env"
os.environ["HF_HUB_OFFLINE"] = "1"  # no test may reach the Hub; code that tries must cope
for _key in ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY", "OPENAI_PROJECT_ID", "ANTHROPIC_API_KEY"):
    os.environ.pop(_key, None)

from sqlalchemy.pool import StaticPool  # noqa: E402
from sqlmodel import Session, create_engine  # noqa: E402

from slm import db  # noqa: E402
from slm.agents import provider  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    db.set_engine(eng)
    yield eng
    provider.set_provider(None)


@pytest.fixture
def client():
    """The API. No `with` block, so the lifespan (and the background worker) doesn't start: jobs just queue."""
    from fastapi.testclient import TestClient

    from slm.api.app import app

    return TestClient(app)


@pytest.fixture
def session(fresh_db):
    with Session(fresh_db) as s:
        yield s


@pytest.fixture
def project(session):
    p = db.Project(name="test", goal="Answer questions about cooking concisely")
    session.add(p)
    session.commit()
    session.refresh(p)
    return p
