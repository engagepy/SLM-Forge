"""What the OpenAI account has spent, read from OpenAI's Costs API.

The figure comes from OpenAI, not from counting tokens. The Costs API needs an organisation
admin key (`OPENAI_ADMIN_KEY`; the project key in OPENAI_API_KEY lacks the `api.usage.read`
scope) and returns daily buckets in USD. The meter narrows it to the OpenAI project the API key
belongs to: found automatically (each project's key list carries redacted key values, whose
last characters identify the key), or set with `OPENAI_PROJECT_ID`. OpenAI updates costs with a
lag of a few hours, so the meter is a read of the account, refreshed every ten minutes, not a
live counter. Other providers: not yet.
"""

import threading
import time
from datetime import UTC, datetime, timedelta

import httpx

from slm.config import get_settings

ORG = "https://api.openai.com/v1/organization"
COSTS_URL = f"{ORG}/costs"
CACHE_SECONDS = 600

_cache: dict = {}
_lock = threading.RLock()  # spend() reads under it and project() may need it again


def _masked(k: str | None) -> str | None:
    if not k:
        return None
    return f"{k[:7]}…{k[-4:]}" if len(k) > 14 else "•••"


def _fetch_costs(admin_key: str, start: datetime, project_id: str | None) -> list[dict]:
    """Daily cost buckets from `start` (UTC) to now, following pagination."""
    buckets, page = [], None
    params: dict = {"start_time": int(start.timestamp()), "bucket_width": "1d", "limit": 180}
    if project_id:
        params["project_ids[]"] = project_id
    with httpx.Client(timeout=20, headers={"Authorization": f"Bearer {admin_key}"}) as client:
        while True:
            r = client.get(COSTS_URL, params=params | ({"page": page} if page else {}))
            if r.status_code != 200:
                try:
                    detail = r.json().get("error", {})
                    detail = detail.get("message", detail) if isinstance(detail, dict) else detail
                except ValueError:
                    detail = r.text[:200]
                raise RuntimeError(f"OpenAI Costs API: {r.status_code}: {detail}")
            body = r.json()
            buckets.extend(body.get("data", []))
            if not body.get("has_more") or not body.get("next_page"):
                return buckets
            page = body["next_page"]


def _paged(client: httpx.Client, url: str) -> list[dict]:
    rows, after = [], None
    while True:
        r = client.get(url, params={"limit": 100} | ({"after": after} if after else {}))
        r.raise_for_status()
        body = r.json()
        rows += body.get("data", [])
        if not body.get("has_more") or not body.get("last_id"):
            return rows
        after = body["last_id"]


def _find_project(admin_key: str, api_key: str) -> dict | None:
    """The OpenAI project the API key belongs to: {"id", "name"}, or None if it can't be told apart."""
    tail = api_key[-4:]
    matches = []
    with httpx.Client(timeout=20, headers={"Authorization": f"Bearer {admin_key}"}) as client:
        for p in _paged(client, f"{ORG}/projects"):
            for k in _paged(client, f"{ORG}/projects/{p['id']}/api_keys"):
                if str(k.get("redacted_value", "")).endswith(tail):
                    matches.append({"id": p["id"], "name": p.get("name", ""), "key_name": k.get("name", "")})
    return matches[0] if len(matches) == 1 else None


def _project_name(admin_key: str, project_id: str) -> str:
    with httpx.Client(timeout=20, headers={"Authorization": f"Bearer {admin_key}"}) as client:
        r = client.get(f"{ORG}/projects/{project_id}")
        return r.json().get("name", "") if r.status_code == 200 else ""


def project() -> dict | None:
    """Which project the meter reads: OPENAI_PROJECT_ID if set, else detected once from the key."""
    s = get_settings()
    if not (s.openai_admin_key and s.openai_api_key):
        return {"id": s.openai_project_id, "name": ""} if s.openai_project_id else None
    with _lock:
        if "project" not in _cache:
            try:
                if s.openai_project_id:
                    _cache["project"] = {
                        "id": s.openai_project_id,
                        "name": _project_name(s.openai_admin_key, s.openai_project_id),
                    }
                else:
                    _cache["project"] = _find_project(s.openai_admin_key, s.openai_api_key)
            except Exception:
                # Organisation-wide then (or the bare id): the reading says which.
                _cache["project"] = {"id": s.openai_project_id, "name": ""} if s.openai_project_id else None
        return _cache["project"]


def _bucket_usd(bucket: dict) -> float:
    return sum(float((res.get("amount") or {}).get("value") or 0) for res in bucket.get("results", []))


def _read(now: datetime) -> dict:
    s = get_settings()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start = min(month_start, now - timedelta(days=30))
    buckets = _fetch_costs(s.openai_admin_key, start, (project() or {}).get("id"))
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    days = {datetime.fromtimestamp(b["start_time"], UTC): _bucket_usd(b) for b in buckets}

    def since(t: datetime) -> float:
        return round(sum(v for d, v in days.items() if d >= t), 2)

    return {
        "today_usd": round(days.get(today, 0.0), 2),
        "month_to_date_usd": since(month_start),
        "last_7_days_usd": since(today - timedelta(days=6)),
        "last_30_days_usd": since(today - timedelta(days=29)),
        "daily": [{"date": d.date().isoformat(), "usd": round(v, 2)} for d, v in sorted(days.items())][-30:],
    }


def _stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def spend(refresh: bool = False) -> dict:
    """The meter's reading, cached for CACHE_SECONDS. Never raises: errors are part of the reading."""
    s = get_settings()
    out = {
        "provider": "openai",
        "configured": bool(s.openai_admin_key),
        "key": _masked(s.openai_api_key),
        "model": s.openai_model,
    }
    if not s.openai_admin_key:
        return out | {"setup": "Set OPENAI_ADMIN_KEY in .env (an organisation admin key from platform.openai.com)."}
    with _lock:
        fresh = _cache.get("at", 0) > time.time() - CACHE_SECONDS
        if not refresh and fresh and "reading" in _cache:
            return out | _cache["reading"]
        if refresh:
            _cache.pop("project", None)  # re-detect too: the key may have moved
        try:
            p = project()
            reading = _read(datetime.now(UTC)) | {
                "as_of": _stamp(),
                "error": None,
                "project_id": p["id"] if p else None,
                "project_name": (p or {}).get("name") or None,
            }
        except Exception as e:  # a bad key, no scope, network: the UI shows why
            reading = {"error": str(e)[:300], "as_of": _stamp()}
        _cache.update(at=time.time(), reading=reading)
        return out | reading
