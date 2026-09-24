"""What the OpenAI account has spent, read from OpenAI's Costs API.

The figure comes from OpenAI, not from counting tokens. The Costs API needs an organisation
admin key (`OPENAI_ADMIN_KEY`; the project key in OPENAI_API_KEY lacks the `api.usage.read`
scope), returns daily buckets in USD, and can be narrowed to one OpenAI project
(`OPENAI_PROJECT_ID`). OpenAI updates it with a lag of a few hours, so the meter is a read of the
account, refreshed every ten minutes, not a live counter. Other providers: not yet.
"""

import threading
import time
from datetime import UTC, datetime, timedelta

import httpx

from slm.config import get_settings

COSTS_URL = "https://api.openai.com/v1/organization/costs"
CACHE_SECONDS = 600

_cache: dict = {}
_lock = threading.Lock()


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


def _bucket_usd(bucket: dict) -> float:
    return sum(float((res.get("amount") or {}).get("value") or 0) for res in bucket.get("results", []))


def _read(now: datetime) -> dict:
    s = get_settings()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start = min(month_start, now - timedelta(days=30))
    buckets = _fetch_costs(s.openai_admin_key, start, s.openai_project_id)
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


def spend(refresh: bool = False) -> dict:
    """The meter's reading, cached for CACHE_SECONDS. Never raises: errors are part of the reading."""
    s = get_settings()
    out = {
        "provider": "openai",
        "configured": bool(s.openai_admin_key),
        "key": _masked(s.openai_api_key),
        "project_id": s.openai_project_id,
        "model": s.openai_model,
    }
    if not s.openai_admin_key:
        return out | {"setup": "Set OPENAI_ADMIN_KEY in .env (an organisation admin key from platform.openai.com)."}
    with _lock:
        fresh = _cache.get("at", 0) > time.time() - CACHE_SECONDS
        if not refresh and fresh and "reading" in _cache:
            return out | _cache["reading"]
        try:
            reading = _read(datetime.now(UTC)) | {
                "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
                "error": None,
            }
        except Exception as e:  # a bad key, no scope, network: the UI shows why
            reading = {"error": str(e)[:300], "as_of": datetime.now(UTC).isoformat(timespec="seconds")}
        _cache.update(at=time.time(), reading=reading)
        return out | reading
