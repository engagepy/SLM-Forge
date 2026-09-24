"""Shared API helpers: sessions, lookups, SSE plumbing."""

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException
from sqlmodel import Session, SQLModel

from slm.db import Project, get_session
from slm.events import bus

SessionDep = Depends(get_session)


def get_or_404[T: SQLModel](s: Session, model: type[T], id_: int) -> T:
    obj = s.get(model, id_)
    if obj is None:
        raise HTTPException(404, f"{model.__name__} {id_} not found")
    return obj


def project_or_404(s: Session, project_id: int) -> Project:
    return get_or_404(s, Project, project_id)


def sse(event: dict) -> dict:
    return {"event": event.get("type", "message"), "data": json.dumps(event, default=str)}


async def topic_stream(topic: str, initial: list[dict] | None = None) -> AsyncIterator[dict]:
    """Forward bus events for `topic` as SSE messages, with a keepalive every 15s."""
    for ev in initial or []:
        yield sse(ev)
    async with bus.subscribe(topic) as q:
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=15)
                yield sse(ev)
            except TimeoutError:
                yield {"event": "ping", "data": "{}"}
