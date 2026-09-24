"""A running meter of what the project spends on the OpenAI API.

OpenAI's billing endpoints need an admin key, which the app doesn't have, so it meters itself:
every call we make returns token counts, we know the model, and the price per token is a setting.
Totals are per project (the Tuner's turns, judging, writing examples) and all-time.
"""

from contextlib import contextmanager
from contextvars import ContextVar

from sqlmodel import Session, func, select

from slm.config import get_settings
from slm.db import ApiUsage, engine
from slm.events import bus

# USD per million tokens, when SLM_OPENAI_PRICE_* isn't set. Check them against your plan.
DEFAULT_PRICES = {"input": 2.0, "cached": 0.5, "output": 8.0}

_scope: ContextVar[tuple[int | None, str]] = ContextVar("usage_scope", default=(None, "other"))


@contextmanager
def scope(project_id: int | None, purpose: str):
    """Attribute every API call made inside to this project and purpose."""
    token = _scope.set((project_id, purpose))
    try:
        yield
    finally:
        _scope.reset(token)


def prices() -> dict:
    s = get_settings()
    return {
        "input": s.openai_price_input if s.openai_price_input is not None else DEFAULT_PRICES["input"],
        "cached": s.openai_price_cached if s.openai_price_cached is not None else DEFAULT_PRICES["cached"],
        "output": s.openai_price_output if s.openai_price_output is not None else DEFAULT_PRICES["output"],
        "configured": s.openai_price_input is not None,
    }


def cost_usd(input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    p = prices()
    fresh = max(0, input_tokens - cached_tokens)
    return (fresh * p["input"] + cached_tokens * p["cached"] + output_tokens * p["output"]) / 1_000_000


def record(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_tokens: int = 0,
    requests: int = 1,
    project_id: int | None = None,
    purpose: str | None = None,
) -> ApiUsage:
    """Store one call (or one agent run) and tell the UI the meter moved."""
    scoped_pid, scoped_purpose = _scope.get()
    row = ApiUsage(
        project_id=scoped_pid if project_id is None else project_id,
        model=model,
        purpose=purpose or scoped_purpose,
        requests=requests,
        input_tokens=int(input_tokens or 0),
        cached_input_tokens=int(cached_tokens or 0),
        output_tokens=int(output_tokens or 0),
        cost_usd=cost_usd(int(input_tokens or 0), int(output_tokens or 0), int(cached_tokens or 0)),
    )
    with Session(engine()) as s:
        s.add(row)
        s.commit()
        s.refresh(row)
    bus.publish("jobs", {"type": "usage", "project_id": row.project_id})
    return row


def masked_key() -> str | None:
    k = get_settings().openai_api_key
    if not k:
        return None
    return f"{k[:7]}…{k[-4:]}" if len(k) > 14 else "•••"


def _totals(s: Session, *where) -> dict:
    row = s.exec(
        select(
            func.coalesce(func.sum(ApiUsage.requests), 0),
            func.coalesce(func.sum(ApiUsage.input_tokens), 0),
            func.coalesce(func.sum(ApiUsage.cached_input_tokens), 0),
            func.coalesce(func.sum(ApiUsage.output_tokens), 0),
            func.coalesce(func.sum(ApiUsage.cost_usd), 0.0),
        ).where(*where)
    ).one()
    return {
        "requests": int(row[0]),
        "input_tokens": int(row[1]),
        "cached_input_tokens": int(row[2]),
        "output_tokens": int(row[3]),
        "cost_usd": round(float(row[4]), 4),
    }


def summary(project_id: int | None = None) -> dict:
    """What the meter reads: this project (if given), all projects, and how the cost is estimated."""
    with Session(engine()) as s:
        out = {"key": masked_key(), "model": get_settings().openai_model, "prices": prices(), "all_time": _totals(s)}
        if project_id is not None:
            out["project"] = _totals(s, ApiUsage.project_id == project_id)
            by = s.exec(
                select(ApiUsage.purpose, func.sum(ApiUsage.cost_usd), func.sum(ApiUsage.requests))
                .where(ApiUsage.project_id == project_id)
                .group_by(ApiUsage.purpose)
            ).all()
            out["project"]["by_purpose"] = {p: {"cost_usd": round(float(c), 4), "requests": int(n)} for p, c, n in by}
    return out
