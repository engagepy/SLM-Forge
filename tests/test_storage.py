"""The disk footprint: everything the app stores locally, by part."""

from slm import storage
from slm.config import get_settings
from slm.db import ModelRecord


def test_footprint_sums_workspace_parts_and_downloaded_models(session, tmp_path, monkeypatch, client):
    s = get_settings()
    s.ensure_dirs()
    (s.datasets_dir / "p1").mkdir(parents=True, exist_ok=True)
    (s.datasets_dir / "p1" / "raw.jsonl").write_bytes(b"x" * 3_000_000)
    (s.exports_dir / "chef").mkdir(parents=True, exist_ok=True)
    (s.exports_dir / "chef" / "weights.safetensors").write_bytes(b"y" * 5_000_000)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_bytes(b"z" * 1_000_000)
    session.add(ModelRecord(repo_id="org/tiny", local_path=str(model)))
    session.commit()
    storage._cache.clear()
    fp = storage.footprint()
    assert fp["parts_gb"]["datasets"] == round(3_000_000 / storage.GB, 3)
    assert fp["parts_gb"]["exports"] == round(5_000_000 / storage.GB, 3)
    assert fp["parts_gb"]["models"] == round(1_000_000 / storage.GB, 3)
    assert fp["total_gb"] == round(9_000_000 / storage.GB, 3) and fp["disk_free_gb"] > 0
    # Cached for a minute, so the header's polling doesn't walk the disk every time.
    (s.exports_dir / "chef" / "more").write_bytes(b"w" * 5_000_000)
    assert storage.footprint()["parts_gb"]["exports"] == fp["parts_gb"]["exports"]
    assert storage.footprint(refresh=True)["parts_gb"]["exports"] > fp["parts_gb"]["exports"]
    assert client.get("/api/system").json()["disk"]["workspace"] == str(s.workspace)
