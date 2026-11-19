"""Build specs (YAML, kind: slm_build) imported as projects."""

from pathlib import Path

import pytest

from slm import spec
from slm.tuner.session import tuner

SAMPLE = (Path(__file__).parent / "fixtures" / "specs" / "legal_clause_extractor.yaml").read_text()


def test_sample_spec_maps_onto_a_project():
    r = spec.load(SAMPLE)
    p = r["project"]
    assert p["name"] == "Legal Clause Extractor 4B"
    assert p["goal"].startswith("A 4B specialist that reads a commercial contract")
    assert "governing_law" in p["system_prompt"] and "verbatim substring" in p["system_prompt"]
    assert '"evidence_span":str|null' in p["system_prompt"]
    assert p["plan"]["source"] == "spec:legal-clause-extractor-4b"
    assert p["plan"]["task_type"] == "extraction"
    assert "theatticusproject/cuad" in p["plan"]["data_sources"]
    assert "baselines." not in p["plan"]["stop_rule"]  # baselines need external models
    assert all(len(v) <= 400 for v in p["plan"].values())


def test_what_the_mac_cannot_do_is_a_warning_not_silence():
    w = " ".join(spec.load(SAMPLE)["warnings"])
    for dropped in (
        "H100",
        "full fine-tuning",
        "15000 rows",
        "awq",
        "never automatic",
        "unattended",
        "8192-token",
        "cuad_to_task_contract",
    ):
        assert dropped in w


def test_system_prompt_is_a_shape_not_the_whole_schema():
    # Regression: the full JSON Schema took 270 of a 512-token training window, leaving too little room for the
    # contract text and the answer; every trained checkpoint scored 0.0 on the legal spec.
    p = spec.load(SAMPLE)["project"]["system_prompt"]
    assert len(p) < 900 and '"type"' not in p and '"properties"' not in p


@pytest.mark.parametrize(
    "text",
    ["not: [valid", "kind: something_else", "kind: slm_build\nmeta: {one_liner: x}", "- a list"],
)
def test_bad_specs_are_refused(text):
    with pytest.raises(ValueError):
        spec.load(text)


def test_from_spec_creates_a_project_but_starts_nothing(client, monkeypatch):
    sent = []
    monkeypatch.setattr(tuner, "send", lambda pid, text, role="user", meta=None: sent.append(text))
    r = client.post("/api/projects/from-spec", json={"yaml": SAMPLE}).json()
    pid = r["project"]["id"]
    assert r["warnings"] and r["project"]["plan"]["source"].startswith("spec:")
    assert sent == [] and client.get(f"/api/projects/{pid}/studio").json()["autopilot"] is False
    assert client.post(f"/api/projects/{pid}/tuner/start").json() == {"started": True}
    assert len(sent) == 1 and "build spec" in sent[0]


def test_from_spec_preview_creates_nothing(client):
    r = client.post("/api/projects/from-spec", json={"yaml": SAMPLE, "preview": True}).json()
    assert r["project"]["name"] == "Legal Clause Extractor 4B" and "id" not in r["project"] and r["warnings"]
    assert client.get("/api/projects").json() == []


def test_from_spec_rejects_an_invalid_spec(client):
    r = client.post("/api/projects/from-spec", json={"yaml": "kind: nope"})
    assert r.status_code == 400 and "slm_build" in r.json()["detail"]
