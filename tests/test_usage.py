"""The spend meter reads OpenAI's Costs API; it never counts tokens."""

import time
from datetime import UTC, datetime, timedelta

from slm import usage
from slm.config import get_settings


def _bucket(day: datetime, value: float, project_id: str | None = None) -> dict:
    return {
        "object": "bucket",
        "start_time": int(day.timestamp()),
        "end_time": int((day + timedelta(days=1)).timestamp()),
        "results": [
            {
                "object": "organization.costs.result",
                "amount": {"value": value, "currency": "usd"},
                "project_id": project_id,
            }
        ],
    }


def test_without_an_admin_key_the_meter_says_how_to_set_it_up(monkeypatch, client):
    monkeypatch.setattr(get_settings(), "openai_admin_key", None)
    out = client.get("/api/usage").json()
    assert out["configured"] is False and "OPENAI_ADMIN_KEY" in out["setup"] and "today_usd" not in out


def test_spend_sums_openai_cost_buckets_for_today_month_and_weeks(monkeypatch, client):
    s = get_settings()
    monkeypatch.setattr(s, "openai_admin_key", "sk-admin-test")
    monkeypatch.setattr(s, "openai_project_id", None)
    monkeypatch.setattr(s, "openai_api_key", "sk-proj-abcdefghijklmnop1234")
    usage._cache.clear()
    # The key's project is found from the admin endpoints, not typed in.
    monkeypatch.setattr(usage, "_find_project", lambda admin, key: {"id": "proj_abc", "name": "SLM Forge"})
    now = datetime.now(UTC)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    calls = []

    def fake_fetch(admin_key, start, project_id):
        calls.append((admin_key, start, project_id))
        return [
            _bucket(today - timedelta(days=d), v) for d, v in ((0, 1.25), (1, 0.5), (6, 2.0), (10, 4.0), (40, 100.0))
        ]

    monkeypatch.setattr(usage, "_fetch_costs", fake_fetch)
    out = client.get("/api/usage").json()
    assert calls[0][0] == "sk-admin-test" and calls[0][2] == "proj_abc"
    assert out["today_usd"] == 1.25 and out["last_7_days_usd"] == 3.75 and out["last_30_days_usd"] == 7.75
    month_start = today.replace(day=1)
    expected_month = sum(
        v for d, v in ((0, 1.25), (1, 0.5), (6, 2.0), (10, 4.0)) if today - timedelta(days=d) >= month_start
    )
    assert out["month_to_date_usd"] == round(expected_month, 2)
    assert out["key"] == "sk-proj…1234" and out["project_id"] == "proj_abc" and out["project_name"] == "SLM Forge"
    assert out["error"] is None
    # Cached: a second read within ten minutes doesn't call OpenAI again; refresh=true does.
    client.get("/api/usage")
    assert len(calls) == 1
    client.get("/api/usage?refresh=true")
    assert len(calls) == 2


def test_an_explicit_project_id_wins_and_gets_its_name(monkeypatch, client):
    s = get_settings()
    monkeypatch.setattr(s, "openai_admin_key", "sk-admin-test")
    monkeypatch.setattr(s, "openai_project_id", "proj_set")
    monkeypatch.setattr(s, "openai_api_key", "sk-proj-abcdefghijklmnop1234")
    usage._cache.clear()
    monkeypatch.setattr(usage, "_find_project", lambda admin, key: (_ for _ in ()).throw(AssertionError("not needed")))
    monkeypatch.setattr(usage, "_project_name", lambda admin, pid: "Named")
    seen = []
    monkeypatch.setattr(usage, "_fetch_costs", lambda admin, start, pid: seen.append(pid) or [])
    out = client.get("/api/usage").json()
    assert seen == ["proj_set"] and out["project_id"] == "proj_set" and out["project_name"] == "Named"


def test_an_ambiguous_or_unlisted_key_falls_back_to_the_whole_organisation(monkeypatch, client):
    s = get_settings()
    monkeypatch.setattr(s, "openai_admin_key", "sk-admin-test")
    monkeypatch.setattr(s, "openai_project_id", None)
    monkeypatch.setattr(s, "openai_api_key", "sk-proj-abcdefghijklmnop1234")
    usage._cache.clear()
    monkeypatch.setattr(usage, "_find_project", lambda admin, key: None)
    seen = []
    monkeypatch.setattr(usage, "_fetch_costs", lambda admin, start, pid: seen.append(pid) or [])
    out = client.get("/api/usage").json()
    assert seen == [None] and out["project_id"] is None and out["month_to_date_usd"] == 0


def test_a_refused_key_shows_as_an_error_not_a_crash(monkeypatch, client):
    monkeypatch.setattr(get_settings(), "openai_admin_key", "sk-admin-bad")
    monkeypatch.setattr(usage, "_find_project", lambda admin, key: None)
    usage._cache.clear()

    def refuse(admin_key, start, project_id):
        raise RuntimeError("OpenAI Costs API: 401: insufficient permissions (api.usage.read)")

    monkeypatch.setattr(usage, "_fetch_costs", refuse)
    out = client.get("/api/usage").json()
    assert out["configured"] is True and "api.usage.read" in out["error"] and "today_usd" not in out
    usage._cache["at"] = time.time() - 10_000  # stale: the next read tries OpenAI again
    monkeypatch.setattr(usage, "_fetch_costs", lambda *a: [])
    assert client.get("/api/usage").json()["error"] is None
