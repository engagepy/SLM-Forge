"""Download base models and read their shape from disk."""

import json
from pathlib import Path

from huggingface_hub import scan_cache_dir, snapshot_download
from sqlmodel import Session, select

from slm import hardware
from slm.config import HF_LOGIN_HELP
from slm.db import ModelRecord, engine
from slm.models.hub import fit_verdict, rough_training_gb

# Everything mlx_lm.load needs; skips PyTorch .bin / ONNX / GGUF duplicates.
ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "*.txt",
    "*.jinja",
    # The licence travels with the weights: an export must be able to pass it on.
    "LICENSE*",
    "LICENCE*",
    "NOTICE*",
    "USE_POLICY*",
    "*.md",
]

GATED_HELP = (
    "{repo} needs you to accept its terms on Hugging Face before it can be downloaded: open "
    "https://huggingface.co/{path}{repo}, accept the licence, then log in: " + HF_LOGIN_HELP + "."
)


def access_error(repo_id: str, e: Exception, repo_type: str = "model") -> Exception:
    """A gated, private or missing repo turned into advice the user can act on; anything else as is."""
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    path = "datasets/" if repo_type == "dataset" else ""
    if isinstance(e, GatedRepoError):
        return RuntimeError(GATED_HELP.format(repo=repo_id, path=path))
    if isinstance(e, RepositoryNotFoundError):
        return RuntimeError(
            f"{repo_id} was not found on Hugging Face, or it is private or gated and you are not logged in "
            f"({HF_LOGIN_HELP})."
        )
    return e


def read_shape(model_path: str | Path) -> hardware.ModelShape:
    with open(Path(model_path) / "config.json") as f:
        return hardware.ModelShape.from_config(json.load(f))


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / hardware.GB


def download(repo_id: str, log=print, local_only: bool = False) -> ModelRecord:
    """Register a model, downloading it into the shared HF cache only if it isn't there yet."""
    origin = "cache"
    try:
        local = Path(snapshot_download(repo_id, allow_patterns=ALLOW_PATTERNS, local_files_only=True))
        log(f"{repo_id} is already on this Mac; no download needed.")
    except Exception:
        if local_only:
            raise
        log(f"Downloading {repo_id} ...")
        try:
            local = Path(snapshot_download(repo_id, allow_patterns=ALLOW_PATTERNS))
        except Exception as e:
            raise access_error(repo_id, e) from e
        origin = "downloaded"
    with open(local / "config.json") as f:
        config = json.load(f)
    shape = hardware.ModelShape.from_config(config)
    size = dir_size_gb(local)
    log(f"Ready at {local} ({size:.2f} GB, ~{shape.params / 1e9:.2f}B params)")

    with Session(engine()) as s:
        rec = s.exec(select(ModelRecord).where(ModelRecord.repo_id == repo_id)).first()
        if rec is None:
            rec = ModelRecord(repo_id=repo_id, local_path=str(local), origin=origin)
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
    """Where a registered model's files are, or None when it was never downloaded or its folder has
    since gone (the HF cache is shared, so another tool may have cleared it): then it isn't on this
    Mac, whatever the row says."""
    with Session(engine()) as s:
        rec = s.exec(select(ModelRecord).where(ModelRecord.repo_id == repo_id)).first()
    if rec and (Path(rec.local_path) / "config.json").exists():
        return rec.local_path
    return None


def serving_path(project) -> str | None:
    """The model a project serves right now: its latest fused model, or else its downloaded base."""
    return project.current_model_path or local_path_for(project.base_model or "")


def local_models() -> list[dict]:
    """Chat/text-generation models already in the Hugging Face cache that MLX can train.

    Vision-language, speech and embedding models are skipped, as are incomplete snapshots."""

    out = []
    try:
        repos = scan_cache_dir().repos
    except Exception:  # no cache yet
        return out
    budget = hardware.detect().budget_gb
    for repo in repos:
        if repo.repo_type != "model" or not repo.revisions:
            continue
        snap = max(repo.revisions, key=lambda r: r.last_modified).snapshot_path
        cfg_path = snap / "config.json"
        if not cfg_path.exists() or not any(snap.glob("*.safetensors")):
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
            archs = cfg.get("architectures") or []
            if "vision_config" in cfg or not any(a.endswith("ForCausalLM") for a in archs):
                continue
            shape = hardware.ModelShape.from_config(cfg)
        except Exception:
            continue

        train_gb = rough_training_gb(shape.params, shape.bits)
        out.append(
            {
                "repo_id": repo.repo_id,
                "path": str(snap),
                "params_b": round(shape.params / 1e9, 2),
                "bits": shape.bits,
                "size_gb": round(repo.size_on_disk / 1e9, 2),
                "train_memory_gb": round(train_gb, 2),
                "fit": fit_verdict(train_gb, budget),
            }
        )
    return sorted(out, key=lambda m: m["params_b"])


def register_local(repo_id: str) -> ModelRecord | None:
    """Register a model that's already in the HF cache, without touching the network."""
    try:
        return download(repo_id, log=lambda _: None, local_only=True)
    except Exception:
        return None
