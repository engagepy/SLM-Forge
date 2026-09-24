"""How much of this Mac's disk the app is using: everything it writes under the workspace, plus
the base models it downloaded into the Hugging Face cache. Sizes are walked, so the reading is
cached for a minute; the header polls it."""

import shutil
import threading
import time
from pathlib import Path

from sqlmodel import Session, select

from slm.config import get_settings
from slm.db import ModelRecord, engine
from slm.hardware import GB

CACHE_SECONDS = 60
_cache: dict = {}
_lock = threading.Lock()


def _size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _measure() -> dict:
    s = get_settings()
    parts = {
        "models": 0,
        "datasets": _size(s.datasets_dir),
        "runs": _size(s.runs_dir),
        "exports": _size(s.exports_dir),
        "uploads": _size(s.uploads_dir),
        "database": sum(_size(p) for p in s.workspace.glob("*.db*")),
    }
    with Session(engine()) as db:
        for rec in db.exec(select(ModelRecord)).all():
            parts["models"] += _size(Path(rec.local_path))  # snapshot links resolve to the cached blobs
    usage = shutil.disk_usage(s.workspace)
    return {  # three decimals: a megabyte, so small parts don't read as nothing
        "total_gb": round(sum(parts.values()) / GB, 3),
        "parts_gb": {k: round(v / GB, 3) for k, v in parts.items()},
        "workspace": str(s.workspace),
        "disk_free_gb": round(usage.free / GB, 1),
        "disk_total_gb": round(usage.total / GB, 1),
    }


def footprint(refresh: bool = False) -> dict:
    with _lock:
        if refresh or _cache.get("at", 0) <= time.time() - CACHE_SECONDS:
            _cache.update(at=time.time(), reading=_measure())
        return _cache["reading"]
