"""Download base models and read their shape from disk."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download
from sqlmodel import Session, select

from slm import hardware
from slm.db import ModelRecord, engine

# Everything mlx_lm.load needs; skips PyTorch .bin / ONNX / GGUF duplicates.
ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "*.txt",
    "*.jinja",
]


def read_shape(model_path: str | Path) -> hardware.ModelShape:
    with open(Path(model_path) / "config.json") as f:
        return hardware.ModelShape.from_config(json.load(f))


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / hardware.GB


def download(repo_id: str, log=print) -> ModelRecord:
    """Download a model into the shared HF cache and register it."""
    log(f"Downloading {repo_id} ...")
    local = Path(snapshot_download(repo_id, allow_patterns=ALLOW_PATTERNS))
    with open(local / "config.json") as f:
        config = json.load(f)
    shape = hardware.ModelShape.from_config(config)
    size = dir_size_gb(local)
    log(f"Downloaded to {local} ({size:.2f} GB, ~{shape.params / 1e9:.2f}B params)")

    with Session(engine()) as s:
        rec = s.exec(select(ModelRecord).where(ModelRecord.repo_id == repo_id)).first()
        if rec is None:
            rec = ModelRecord(repo_id=repo_id, local_path=str(local))
        rec.local_path = str(local)
        rec.params = shape.params
        rec.bits = shape.bits
        rec.size_gb = round(size, 3)
        rec.config = config
        s.add(rec)
        s.commit()
        s.refresh(rec)
        return rec


def local_path_for(repo_id: str) -> str | None:
    with Session(engine()) as s:
        rec = s.exec(select(ModelRecord).where(ModelRecord.repo_id == repo_id)).first()
        return rec.local_path if rec else None
