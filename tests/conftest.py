import os
import tempfile

import pytest

# Isolate every test run from the real workspace before slm modules read settings.
os.environ["SLM_WORKSPACE"] = tempfile.mkdtemp(prefix="slm-test-")

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
