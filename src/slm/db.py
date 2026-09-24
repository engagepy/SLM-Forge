"""SQLite persistence. Every artifact records the project and parent it came from."""

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, Session, SQLModel, create_engine

from slm.config import get_settings


def now() -> datetime:
    return datetime.now(UTC)


def json_field(default: Any = None) -> Any:
    factory = (lambda: {}) if default is None else (lambda: default.copy())
    return Field(default_factory=factory, sa_column=Column(JSON))


class Project(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    goal: str = ""
    base_model: str | None = None  # HF repo id
    system_prompt: str = ""
    # The model the project currently serves: base path, or latest fused model.
    current_model_path: str | None = None
    current_adapter_path: str | None = None
    created_at: datetime = Field(default_factory=now)


class ModelRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    repo_id: str = Field(index=True, unique=True)
    local_path: str
    params: int = 0
    bits: float = 16
    size_gb: float = 0
    config: dict = json_field()
    downloaded_at: datetime = Field(default_factory=now)


class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    name: str
    source: str  # hf | upload | feedback | synthetic
    source_ref: str = ""  # repo id, filename, ...
    license: str = ""
    raw_path: str = ""
    n_rows: int = 0
    columns: list = json_field([])
    created_at: datetime = Field(default_factory=now)


class DatasetVersion(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    dataset_id: int | None = Field(default=None, foreign_key="dataset.id")
    kind: str = "sft"  # sft | dpo
    path: str  # directory holding train/valid/test.jsonl
    n_train: int = 0
    n_valid: int = 0
    n_test: int = 0
    mapping: dict = json_field()
    cleaning_report: dict = json_field()
    token_stats: dict = json_field()
    created_at: datetime = Field(default_factory=now)


class Job(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int | None = Field(default=None, foreign_key="project.id", index=True)
    kind: str  # sft | dpo | fuse | download | export
    status: str = "queued"  # queued | running | succeeded | failed | cancelled
    config: dict = json_field()
    result: dict = json_field()
    log_path: str = ""
    error: str = ""
    created_at: datetime = Field(default_factory=now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class Metric(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="job.id", index=True)
    iteration: int
    split: str  # train | val
    values: dict = json_field()


class Checkpoint(SQLModel, table=True):
    """A trained model state: an adapter, optionally fused into a standalone model."""

    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    job_id: int | None = Field(default=None, foreign_key="job.id")
    parent_id: int | None = Field(default=None, foreign_key="checkpoint.id")
    kind: str  # sft | dpo
    base_model_path: str
    adapter_path: str
    fused_path: str | None = None
    metrics: dict = json_field()
    created_at: datetime = Field(default_factory=now)


class Feedback(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    prompt: str
    system: str = ""
    candidate_a: str
    candidate_b: str
    params_a: dict = json_field()
    params_b: dict = json_field()
    choice: str  # a | b | tie | both_bad
    edited_answer: str = ""
    critique: str = ""
    model_ref: str = ""
    observed: bool = False  # processed by the Observer
    created_at: datetime = Field(default_factory=now)


class PreferencePair(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    prompt: str
    system: str = ""
    chosen: str
    rejected: str
    source: str = "human"  # human | synthetic
    feedback_id: int | None = Field(default=None, foreign_key="feedback.id")
    approved: bool = True
    used_in_job_id: int | None = None
    created_at: datetime = Field(default_factory=now)


class SftExample(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    messages: list = json_field([])
    source: str = "feedback"  # feedback | synthetic
    feedback_id: int | None = Field(default=None, foreign_key="feedback.id")
    approved: bool = True
    used_in_job_id: int | None = None
    created_at: datetime = Field(default_factory=now)


class Proposal(SQLModel, table=True):
    """An action an agent wants to take. Nothing costly runs until a human approves it."""

    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    agent: str
    action: str  # import_dataset | acquire_manually | run_sft | run_dpo | generate_synthetic | ...
    title: str
    rationale: str = ""
    payload: dict = json_field()
    status: str = "pending"  # pending | approved | rejected | executed | failed
    result: dict = json_field()
    created_at: datetime = Field(default_factory=now)
    decided_at: datetime | None = None


class AgentEvent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int | None = Field(default=None, foreign_key="project.id", index=True)
    agent: str
    kind: str  # message | tool_call | tool_result | proposal | error
    content: dict = json_field()
    created_at: datetime = Field(default_factory=now)


class TunerMessage(SQLModel, table=True):
    """The visible Studio transcript. (The agent's own memory lives in its SDK session.)"""

    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    role: str  # user | assistant | event | tool
    content: str = ""
    meta: dict = json_field()  # tool name/args/summary, event details
    created_at: datetime = Field(default_factory=now)


class StudioState(SQLModel, table=True):
    """What the Studio canvas is focused on, as set by the Tuner."""

    project_id: int = Field(foreign_key="project.id", primary_key=True)
    stage: str = "goal"  # goal | model | data | train | evaluate | refine | export
    note: str = ""  # one line the Tuner wants on the canvas right now
    comparisons: list = json_field([])  # A/B tasks waiting for the human
    samples: list = json_field([])  # recent model outputs the Tuner showed
    autopilot: bool = True  # the Tuner keeps going on its own until the model is exported
    completed: bool = False  # set by the Tuner's finish_project
    stalled_nudges: int = 0  # autopilot nudges in a row that made no progress
    updated_at: datetime = Field(default_factory=now)


class UserProfile(SQLModel, table=True):
    """What the Tuner has learned about the person using this Mac, across all their projects."""

    id: int | None = Field(default=None, primary_key=True)  # one local user: id 1
    level: str = "unknown"  # unknown | beginner | intermediate | expert
    level_evidence: str = ""  # why the Tuner thinks so, in its words
    notes: list = json_field([])  # [{id, kind, text, project_id, created_at}]
    updated_at: datetime = Field(default_factory=now)


_engine = None


def engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        settings.ensure_dirs()
        _engine = create_engine(
            f"sqlite:///{settings.db_path}",
            connect_args={"check_same_thread": False},
        )
        SQLModel.metadata.create_all(_engine)
        _add_missing_columns(_engine)
    return _engine


def _add_missing_columns(eng) -> None:
    """create_all() makes new tables but never alters existing ones. Add columns introduced after
    a table was first created (additive only, with the model's default), so older workspaces keep
    working without a migration tool."""
    from sqlalchemy import inspect, text

    insp = inspect(eng)
    with eng.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing:
                    continue
                default = col.default.arg if col.default is not None and not callable(col.default.arg) else None
                sql_default = (
                    ""
                    if default is None
                    else f" DEFAULT {int(default) if isinstance(default, bool) else repr(default)}"
                )
                col_type = col.type.compile(eng.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col_type}{sql_default}'))


def set_engine(new_engine) -> None:
    """Swap the engine (tests use an in-memory database)."""
    global _engine
    _engine = new_engine
    SQLModel.metadata.create_all(_engine)


def get_session() -> Iterator[Session]:
    with Session(engine()) as session:
        yield session


def log_event(project_id: int | None, agent: str, kind: str, content: dict) -> None:
    from slm.events import bus

    with Session(engine()) as s:
        ev = AgentEvent(project_id=project_id, agent=agent, kind=kind, content=content)
        s.add(ev)
        s.commit()
        s.refresh(ev)
        if project_id is not None:
            bus.publish(f"project:{project_id}", {"type": "agent_event", **ev.model_dump(mode="json")})
