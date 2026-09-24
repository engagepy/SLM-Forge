"""Additive migration: columns added after a table was created must not leave NULLs behind."""

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from slm import db


def test_json_columns_added_later_are_backfilled_with_their_defaults(tmp_path):
    # Regression: StudioState.evals / granted came back NULL on older workspaces, so the Studio
    # snapshot crashed on `st.evals[-12:]` and confirming a proposal on `[*st.granted, ...]`.
    eng = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    SQLModel.metadata.create_all(eng)
    with eng.begin() as conn:
        for col in ("evals", "granted", "pending_action"):
            conn.execute(text(f'ALTER TABLE studiostate DROP COLUMN "{col}"'))
        conn.execute(text("ALTER TABLE project DROP COLUMN test_questions"))
        conn.execute(
            text(
                "INSERT INTO project (id, name, goal, system_prompt, created_at) VALUES (1, 'old', '', '', '2026-01-01')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO studiostate (project_id, stage, note, comparisons, samples, autopilot, completed, "
                "stalled_nudges, updated_at) VALUES (1, 'goal', '', '[]', '[]', 0, 0, 0, '2026-01-01')"
            )
        )
    db._add_missing_columns(eng)
    # A workspace migrated before the backfill existed: the column is there, but NULL.
    with eng.begin() as conn:
        conn.execute(text("UPDATE studiostate SET granted = NULL"))
    db._add_missing_columns(eng)
    with Session(eng) as s:
        st = s.get(db.StudioState, 1)
        assert st.evals == [] and st.granted == [] and st.pending_action == {}
        assert s.get(db.Project, 1).test_questions == []
