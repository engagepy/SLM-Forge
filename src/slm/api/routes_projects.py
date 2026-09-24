"""Projects, datasets, training runs, checkpoints and exports."""

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from slm import hardware, storage
from slm.agents.base import set_proposal_status
from slm.api.common import SessionDep, get_or_404, project_or_404
from slm.config import get_settings
from slm.data import format as fmt
from slm.data import pipeline, scout_tools
from slm.db import (
    Checkpoint,
    Dataset,
    DatasetVersion,
    Feedback,
    PreferencePair,
    Project,
    Proposal,
    SftExample,
    awaiting_review,
    count,
    ready,
)
from slm.models import manage
from slm.sessions import export_jobs, on_disk
from slm.train.config import TrainConfig, preset
from slm.train.jobs import serve_checkpoint
from slm.train.worker import worker

router = APIRouter(prefix="/api/projects", tags=["projects"])

# ── projects ────────────────────────────────────────────────────────────────


class ProjectIn(BaseModel):
    name: str
    goal: str = ""
    base_model: str | None = None
    system_prompt: str = ""


class ProjectPatch(BaseModel):
    name: str | None = None
    goal: str | None = None
    base_model: str | None = None
    system_prompt: str | None = None


@router.get("")
def list_projects(s: Session = SessionDep) -> list[dict]:
    return [p.model_dump(mode="json") for p in s.exec(select(Project).order_by(Project.id.desc())).all()]


@router.post("")
def create_project(body: ProjectIn, s: Session = SessionDep) -> dict:
    p = Project(**body.model_dump())
    if p.base_model and (path := manage.local_path_for(p.base_model)):
        p.current_model_path = path
    s.add(p)
    s.commit()
    s.refresh(p)
    return p.model_dump(mode="json")


@router.patch("/{project_id}")
def update_project(project_id: int, body: ProjectPatch, s: Session = SessionDep) -> dict:
    p = project_or_404(s, project_id)
    changes = body.model_dump(exclude_none=True)
    if "base_model" in changes and changes["base_model"] != p.base_model:
        if s.exec(select(Checkpoint).where(Checkpoint.project_id == p.id)).first():
            raise HTTPException(409, "Base model can't change after training has started; create a new project.")
        p.current_model_path = manage.local_path_for(changes["base_model"])
        p.current_adapter_path = None
    for k, v in changes.items():
        setattr(p, k, v)
    s.add(p)
    s.commit()
    s.refresh(p)
    return p.model_dump(mode="json")


@router.get("/{project_id}")
def project_overview(project_id: int, s: Session = SessionDep) -> dict:
    p = project_or_404(s, project_id)
    checkpoints = s.exec(select(Checkpoint).where(Checkpoint.project_id == project_id).order_by(Checkpoint.id)).all()
    return {
        "project": p.model_dump(mode="json"),
        "base_model_downloaded": bool(p.base_model and manage.local_path_for(p.base_model)),
        "counts": {
            "datasets": count(s, Dataset, Dataset.project_id == project_id),
            "dataset_versions": count(s, DatasetVersion, DatasetVersion.project_id == project_id),
            "feedback": count(s, Feedback, Feedback.project_id == project_id),
            "pairs_ready": count(s, PreferencePair, PreferencePair.project_id == project_id, *ready(PreferencePair)),
            "sft_ready": count(s, SftExample, SftExample.project_id == project_id, *ready(SftExample)),
            "awaiting_review": sum(
                count(s, m, m.project_id == project_id, *awaiting_review(m)) for m in (PreferencePair, SftExample)
            ),
            "pending_proposals": count(s, Proposal, Proposal.project_id == project_id, Proposal.status == "pending"),
        },
        "checkpoints": [c.model_dump(mode="json") for c in checkpoints],
    }


# ── datasets ────────────────────────────────────────────────────────────────


@router.get("/{project_id}/datasets")
def list_datasets(project_id: int, s: Session = SessionDep) -> dict:
    project_or_404(s, project_id)
    ds = s.exec(select(Dataset).where(Dataset.project_id == project_id).order_by(Dataset.id.desc())).all()
    vs = s.exec(
        select(DatasetVersion).where(DatasetVersion.project_id == project_id).order_by(DatasetVersion.id.desc())
    ).all()
    return {
        "datasets": [d.model_dump(mode="json") | {"suggested_mapping": fmt.guess_mapping(d.columns)} for d in ds],
        "versions": [v.model_dump(mode="json") for v in vs],
    }


@router.post("/{project_id}/datasets/upload")
def upload_dataset(
    project_id: int,
    file: UploadFile = File(...),
    name: str = Form(""),
    proposal_id: int | None = Form(None),
    s: Session = SessionDep,
) -> dict:
    project_or_404(s, project_id)
    settings = get_settings()
    safe_name = Path(file.filename or "upload").name
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    tmp = settings.uploads_dir / f"p{project_id}-{safe_name}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    label = name or Path(safe_name).stem
    dest = settings.datasets_dir / f"p{project_id}" / "raw" / f"{label}-upload{tmp.stat().st_mtime_ns}"
    try:
        n, columns = scout_tools.import_upload(tmp, dest)
    except (ValueError, UnicodeDecodeError) as e:
        raise HTTPException(400, f"Could not read {safe_name}: {e}") from e
    ds = Dataset(
        project_id=project_id,
        name=label,
        source="upload",
        source_ref=safe_name,
        raw_path=str(dest / "raw.jsonl"),
        n_rows=n,
        columns=columns,
    )
    s.add(ds)
    s.commit()
    s.refresh(ds)
    if proposal_id:
        # This upload fulfils a guided-acquisition proposal.

        set_proposal_status(proposal_id, "executed", {"dataset_id": ds.id})
    return ds.model_dump(mode="json") | {"suggested_mapping": fmt.guess_mapping(columns)}


class MappingIn(BaseModel):
    mapping: dict


@router.post("/{project_id}/datasets/{dataset_id}/mapping-preview")
def mapping_preview(project_id: int, dataset_id: int, body: MappingIn, s: Session = SessionDep) -> dict:
    ds = get_or_404(s, Dataset, dataset_id)
    return pipeline.preview_mapping(ds, body.mapping, n=5)


class PrepareIn(BaseModel):
    mapping: dict
    rules: dict = {}
    max_seq_length: int = 1024
    valid_frac: float = 0.05
    test_frac: float = 0.05
    include_feedback: bool = False


@router.post("/{project_id}/datasets/{dataset_id}/prepare")
def prepare_dataset(project_id: int, dataset_id: int, body: PrepareIn, s: Session = SessionDep) -> dict:
    get_or_404(s, Dataset, dataset_id)
    job = worker.submit("prepare_dataset", {"dataset_id": dataset_id, **body.model_dump()}, project_id)
    return {"job_id": job.id}


@router.get("/{project_id}/versions/{version_id}/sample")
def version_sample(
    project_id: int, version_id: int, split: str = "train", limit: int = 10, s: Session = SessionDep
) -> dict:
    v = get_or_404(s, DatasetVersion, version_id)
    return {"version": v.model_dump(mode="json"), "records": pipeline.read_split(v, split, min(limit, 100))}


# ── training ────────────────────────────────────────────────────────────────


def _model_path(p: Project) -> str:
    path = manage.serving_path(p)
    if not path:
        raise HTTPException(409, "Download the project's base model first")
    return path


def _typical_len(s: Session, version_id: int | None) -> int | None:
    """p95 token length of a dataset version: what batches actually pad to."""
    if not version_id:
        return None
    v = s.get(DatasetVersion, version_id)
    return (v.token_stats or {}).get("p95") if v else None


@router.get("/{project_id}/train/presets")
def train_presets(
    project_id: int, mode: str = "sft", dataset_version_id: int | None = None, s: Session = SessionDep
) -> dict:
    p = project_or_404(s, project_id)
    shape = manage.read_shape(_model_path(p))
    typical = _typical_len(s, dataset_version_id)
    out = {}
    for name in ("safe", "balanced", "quality"):
        cfg = preset(name, shape, mode, typical)
        out[name] = {"config": cfg.model_dump(), "estimate": cfg.memory_estimate(shape, typical).to_dict()}
    return {"presets": out, "budget_gb": hardware.detect().budget_gb, "typical_len": typical}


class EstimateIn(BaseModel):
    train: TrainConfig
    dataset_version_id: int | None = None


@router.post("/{project_id}/train/estimate")
def train_estimate(project_id: int, body: EstimateIn, s: Session = SessionDep) -> dict:
    p = project_or_404(s, project_id)
    shape = manage.read_shape(_model_path(p))
    typical = _typical_len(s, body.dataset_version_id)
    out = {"estimate": body.train.memory_estimate(shape, typical).to_dict(), "typical_len": typical}
    if body.dataset_version_id and (v := s.get(DatasetVersion, body.dataset_version_id)):
        out |= {"iters": body.train.total_iters(v.n_train), "epochs": body.train.epochs_for(v.n_train)}
    return out


class TrainIn(BaseModel):
    train: TrainConfig
    dataset_version_id: int | None = None
    start_from: str = "current"  # current (continue the lineage) | base (fresh run)
    force: bool = False  # launch even if the estimate says it won't fit


def _launch(project_id: int, kind: str, body: TrainIn, s: Session) -> dict:
    p = project_or_404(s, project_id)
    shape = manage.read_shape(_model_path(p))
    est = body.train.memory_estimate(shape, _typical_len(s, body.dataset_version_id))
    if not est.fits and not body.force:
        raise HTTPException(
            422,
            f"Estimated {est.total_gb:.1f} GB exceeds the {est.budget_gb:.1f} GB budget. "
            "Lower batch size / sequence length, enable gradient checkpointing, or pass force.",
        )
    cfg = {"train": body.train.model_dump(), "start_from": body.start_from}
    if body.dataset_version_id:
        v = get_or_404(s, DatasetVersion, body.dataset_version_id)
        if kind == "sft" and v.kind != "sft":
            raise HTTPException(422, "SFT needs an SFT dataset version")
        cfg["dataset_version_id"] = v.id
    elif kind == "sft":
        raise HTTPException(422, "dataset_version_id is required for SFT")
    job = worker.submit(kind, cfg, project_id)
    return {"job_id": job.id, "estimate": est.to_dict()}


@router.post("/{project_id}/train/sft")
def train_sft(project_id: int, body: TrainIn, s: Session = SessionDep) -> dict:
    body.train.mode = "sft"
    return _launch(project_id, "sft", body, s)


@router.post("/{project_id}/train/dpo")
def train_dpo(project_id: int, body: TrainIn, s: Session = SessionDep) -> dict:
    body.train.mode = "dpo"
    return _launch(project_id, "dpo", body, s)


# ── checkpoints & export ────────────────────────────────────────────────────


@router.post("/{project_id}/checkpoints/{checkpoint_id}/activate")
def activate_checkpoint(project_id: int, checkpoint_id: int, s: Session = SessionDep) -> dict:
    """Serve an earlier checkpoint (roll back). Later training continues from it."""
    p = project_or_404(s, project_id)
    c = get_or_404(s, Checkpoint, checkpoint_id)
    if c.project_id != p.id:
        raise HTTPException(404)
    serve_checkpoint(s, p, c)
    return p.model_dump(mode="json")


class ExportIn(BaseModel):
    name: str = ""
    quantize_bits: int | None = None
    sampling: dict = {}


@router.post("/{project_id}/export")
def export_model(project_id: int, body: ExportIn, s: Session = SessionDep) -> dict:
    p = project_or_404(s, project_id)
    _model_path(p)
    if body.quantize_bits not in (None, 3, 4, 6, 8):
        raise HTTPException(422, "quantize_bits must be 3, 4, 6 or 8")
    return {"job_id": worker.submit("export", body.model_dump(), project_id).id}


@router.delete("/{project_id}")
def delete_project(project_id: int, keep_exports: bool = False, s: Session = SessionDep) -> dict:
    """Remove the project, its runs, data, chat and memory; its exported models too unless kept."""
    project_or_404(s, project_id)
    s.close()
    return storage.delete_project(project_id, keep_exports=keep_exports)


@router.delete("/{project_id}/exports/{job_id}")
def delete_export(project_id: int, job_id: int, s: Session = SessionDep) -> dict:
    project_or_404(s, project_id)
    try:
        return storage.delete_export(project_id, job_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/{project_id}/exports")
def list_exports(project_id: int, s: Session = SessionDep) -> list[dict]:
    return [
        {"job_id": j.id, **j.result, "name": Path(j.result.get("path", "")).name, "on_disk": on_disk(j),
         "created_at": j.finished_at}
        for j in export_jobs(s, project_id)
    ]  # fmt: skip
