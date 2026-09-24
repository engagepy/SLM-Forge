"""How much of this Mac's disk the app is using, item by item, and how to give it back.

Everything the app writes lives under the workspace (datasets, training runs, exported models,
uploads, the databases); the base models it downloads go into the Hugging Face cache, a folder
shared by every program on the Mac that loads Hugging Face models. Sizes are walked, so the
footprint is cached for a minute; every deletion refreshes it and reports what it freed.
"""

import shutil
import threading
import time
from pathlib import Path

from sqlmodel import Session, delete, select

from slm.config import get_settings
from slm.db import (
    AgentEvent,
    Checkpoint,
    Dataset,
    DatasetVersion,
    Feedback,
    Job,
    Metric,
    ModelRecord,
    PreferencePair,
    Project,
    Proposal,
    SftExample,
    StudioState,
    TunerMessage,
    engine,
)
from slm.hardware import GB
from slm.sessions import export_jobs

CACHE_SECONDS = 60
_cache: dict = {}
_lock = threading.Lock()


def _size(path: Path | str | None) -> int:
    p = Path(path) if path else None
    if p is None or not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _gb(n: int) -> float:
    return round(n / GB, 3)  # a megabyte: small parts shouldn't read as nothing


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
            parts["models"] += _size(rec.local_path)  # snapshot links resolve to the cached blobs
    usage = shutil.disk_usage(s.workspace)
    total = sum(parts.values())
    reclaimable = sum(item["bytes"] for item in reclaimable_items())
    return {
        "total_gb": _gb(total),
        "parts_gb": {k: _gb(v) for k, v in parts.items()},
        "workspace": str(s.workspace),
        "disk_free_gb": round(usage.free / GB, 1),
        "disk_total_gb": round(usage.total / GB, 1),
        "reclaimable_gb": _gb(reclaimable),
        "tidy_threshold_gb": s.disk_tidy_gb,
        "tidy_suggested": total / GB > s.disk_tidy_gb and reclaimable / GB >= 0.5,
    }


# ── what earlier runs left behind ───────────────────────────────────────────

FUSED_DIRS = ("base-fused", "policy-fused", "fused")


def reclaimable_items() -> list[dict]:
    """Run files nothing depends on any more: the fused model copies a run wrote so the next run
    could start from it (rebuildable from base + adapter, which stay), and the folders of runs that
    failed or were cancelled. A fused copy is kept while any project serves it or any checkpoint
    was trained on top of it, so rolling back to any checkpoint still works after tidying."""
    cfg = get_settings()
    with Session(engine()) as s:
        needed = {p.current_model_path for p in s.exec(select(Project)).all() if p.current_model_path}
        needed |= {c.base_model_path for c in s.exec(select(Checkpoint)).all()}
        checkpoint_jobs = {c.job_id for c in s.exec(select(Checkpoint)).all()}
        dead_jobs = [
            j.id
            for j in s.exec(select(Job).where(Job.status.in_(["failed", "cancelled"]))).all()
            if j.id not in checkpoint_jobs
        ]
    items, whole = [], set()
    for job_id in dead_jobs:
        run = cfg.runs_dir / f"job-{job_id:05d}"
        if run.is_dir() and (n := _size(run)):
            items.append({"path": str(run), "bytes": n, "why": f"run {job_id} failed or was cancelled"})
            whole.add(run)
    for run in sorted(cfg.runs_dir.glob("job-*")) if cfg.runs_dir.exists() else []:
        if run in whole:
            continue  # the whole folder is already listed
        for name in FUSED_DIRS:
            fused = run / name
            if fused.is_dir() and str(fused) not in needed and (n := _size(fused)):
                items.append(
                    {"path": str(fused), "bytes": n, "why": f"{run.name}: fused copy, rebuildable from its adapter"}
                )
    return items


def tidy() -> dict:
    """Delete what reclaimable_items lists, and forget fused paths that no longer exist."""
    items = reclaimable_items()
    freed = sum(_rmtree(item["path"]) for item in items)
    with Session(engine()) as s:
        for c in s.exec(select(Checkpoint)).all():
            if c.fused_path and not Path(c.fused_path).exists():
                c.fused_path = None  # serve_checkpoint falls back to base + adapter
                s.add(c)
        s.commit()
    footprint(refresh=True)
    return {"freed_gb": _gb(freed), "deleted": len(items)}


def footprint(refresh: bool = False) -> dict:
    with _lock:
        if refresh or _cache.get("at", 0) <= time.time() - CACHE_SECONDS:
            _cache.update(at=time.time(), reading=_measure())
        return _cache["reading"]


# ── the inventory ───────────────────────────────────────────────────────────


def _project_files(s: Session, project_id: int) -> dict:
    """Every path a project owns on disk, by part."""
    cfg = get_settings()
    job_ids = [j.id for j in s.exec(select(Job).where(Job.project_id == project_id)).all()]
    return {
        "runs": [cfg.runs_dir / f"job-{i:05d}" for i in job_ids],
        "datasets": [cfg.datasets_dir / f"p{project_id}"],
        "uploads": list(cfg.uploads_dir.glob(f"p{project_id}-*")),
    }


def inventory() -> dict:
    """What's on disk because of this app: models, projects (with their exports), and the footprint."""
    with Session(engine()) as s:
        projects = s.exec(select(Project).order_by(Project.id.desc())).all()
        used_by: dict[str, list[dict]] = {}
        for p in projects:
            if p.base_model:
                used_by.setdefault(p.base_model, []).append({"id": p.id, "name": p.name})
        models = [
            {
                "repo_id": m.repo_id,
                "size_gb": _gb(_size(m.local_path)),
                "origin": m.origin,
                "used_by": used_by.get(m.repo_id, []),
                "downloaded_at": m.downloaded_at.isoformat(),
            }
            for m in s.exec(select(ModelRecord).order_by(ModelRecord.downloaded_at.desc())).all()
        ]
        rows = []
        for p in projects:
            files = _project_files(s, p.id)
            exports = [
                {
                    "job_id": j.id,
                    "name": Path(j.result["path"]).name,
                    "path": j.result["path"],
                    "size_gb": _gb(_size(j.result["path"])),
                }
                for j in export_jobs(s, p.id)
                if Path(j.result.get("path", "")).is_dir()
            ]
            parts = {k: _gb(sum(_size(x) for x in v)) for k, v in files.items()}
            parts["exports"] = round(sum(e["size_gb"] for e in exports), 3)
            rows.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "goal": p.goal,
                    "base_model": p.base_model,
                    "parts_gb": parts,
                    "total_gb": round(sum(parts.values()), 3),
                    "exports": exports,
                }
            )
    return {"models": models, "projects": rows, "footprint": footprint()}


# ── giving space back ───────────────────────────────────────────────────────


def _rmtree(path: Path | str | None) -> int:
    """Delete a folder (or file) and return the bytes it held."""
    p = Path(path) if path else None
    if p is None or not p.exists():
        return 0
    n = _size(p)
    shutil.rmtree(p) if p.is_dir() else p.unlink()
    return n


def _delete_from_hf_cache(repo_id: str, local_path: str) -> int:
    """Remove a model from the Hugging Face cache; returns bytes freed."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        repo = next((r for r in info.repos if r.repo_id == repo_id and r.repo_type == "model"), None)
        if repo is not None:
            strategy = info.delete_revisions(*[rev.commit_hash for rev in repo.revisions])
            strategy.execute()
            return int(strategy.expected_freed_size)
    except Exception:
        pass  # not in the cache (or no cache): fall through to the folder itself
    # A snapshot folder of symlinks frees nothing by itself: count only the bytes that really go.
    p = Path(local_path)
    real = sum(f.lstat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) if p.exists() else 0
    _rmtree(local_path)
    return real


def remove_model(repo_id: str) -> dict:
    """Delete a downloaded base model, unless a project still builds on it."""
    with Session(engine()) as s:
        rec = s.exec(select(ModelRecord).where(ModelRecord.repo_id == repo_id)).first()
        if rec is None:
            raise LookupError(f"{repo_id} isn't registered")
        users = s.exec(select(Project.name).where(Project.base_model == repo_id)).all()
        if users:
            raise ValueError(f"{repo_id} is the base model of {', '.join(users)}; delete those projects first.")
        local_path = rec.local_path
    from slm.inference.engine import engine as infer

    if (infer.loaded or {}).get("model_path") == local_path:
        infer.unload()
    # Files first: if the delete fails the row stays, so the model doesn't vanish from the app while
    # its gigabytes stay on disk.
    freed = _delete_from_hf_cache(repo_id, local_path)
    if Path(local_path).exists():
        raise RuntimeError(f"{local_path} is still there after the delete")
    with Session(engine()) as s:
        rec = s.exec(select(ModelRecord).where(ModelRecord.repo_id == repo_id)).first()
        if rec is not None:
            s.delete(rec)
            s.commit()
    footprint(refresh=True)
    return {"repo_id": repo_id, "freed_gb": _gb(freed)}


def delete_export(project_id: int, job_id: int) -> dict:
    with Session(engine()) as s:
        job = s.get(Job, job_id)
        if job is None or job.project_id != project_id or job.kind != "export":
            raise LookupError("no such export in this project")
        path = (job.result or {}).get("path")
        job.result = dict(job.result or {}) | {"deleted": True}
        s.add(job)
        s.commit()
    freed = _rmtree(path)
    footprint(refresh=True)
    return {"job_id": job_id, "freed_gb": _gb(freed)}


def reset_project(project_id: int, keep_export_job_ids: list[int]) -> dict:
    """Wipe a project's history but keep the project and the chosen exported models: runs, data,
    checkpoints, chat and Tuner memory go; the goal, system prompt, plan and test set stay, so the
    next round starts clean. The base model is unset (a fresh round picks one again)."""
    from slm.sessions import stop_project
    from slm.tuner.session import tuner

    stop_project(project_id, cancel_jobs=True)
    keep = set(keep_export_job_ids)
    with Session(engine()) as s:
        project = s.get(Project, project_id)
        if project is None:
            raise LookupError(f"project {project_id} not found")
        files = _project_files(s, project_id)
        export_paths, kept = [], []
        for j in export_jobs(s, project_id):
            (kept if j.id in keep else export_paths).append(j.result.get("path"))
        kept_runs = {cfg_run for cfg_run in ()}  # export folders live under exports/, not runs/
        job_ids = [j.id for j in s.exec(select(Job).where(Job.project_id == project_id)).all() if j.id not in keep]
        s.exec(delete(Metric).where(Metric.job_id.in_(job_ids)))
        for model in (Checkpoint, PreferencePair, SftExample, Feedback, DatasetVersion, Dataset, Proposal,
                      AgentEvent, TunerMessage):  # fmt: skip
            s.exec(delete(model).where(model.project_id == project_id))
        s.exec(delete(Job).where(Job.id.in_(job_ids)))
        st = s.get(StudioState, project_id)
        if st is not None:
            st.stage, st.note, st.comparisons, st.samples, st.evals = "goal", "", [], [], []
            st.autopilot, st.completed, st.stalled_nudges, st.pending_action, st.granted = False, False, 0, {}, []
            s.add(st)
        project.base_model = project.current_model_path = project.current_adapter_path = None
        s.add(project)
        s.commit()
    from slm.inference.engine import engine as infer

    infer.unload()
    freed = sum(_rmtree(p) for paths in files.values() for p in paths) + sum(_rmtree(p) for p in export_paths)
    del kept_runs
    tuner.forget(project_id)
    footprint(refresh=True)
    return {"project_id": project_id, "freed_gb": _gb(freed), "kept_exports": [Path(p).name for p in kept]}


def delete_project(project_id: int, keep_exports: bool = False) -> dict:
    """Remove a project: its Tuner, jobs, data, runs, memory and (unless kept) its exported models.
    Downloaded base models stay: they may serve other projects, and are removed separately."""
    from slm.sessions import stop_project
    from slm.tuner.session import tuner

    stop_project(project_id, cancel_jobs=True)  # halts the Tuner and cancels running/queued jobs
    with Session(engine()) as s:
        project = s.get(Project, project_id)
        if project is None:
            raise LookupError(f"project {project_id} not found")
        files = _project_files(s, project_id)
        export_paths = [] if keep_exports else [j.result.get("path") for j in export_jobs(s, project_id)]
        job_ids = list(s.exec(select(Job.id).where(Job.project_id == project_id)).all())
        s.exec(delete(Metric).where(Metric.job_id.in_(job_ids)))
        for model in (Checkpoint, PreferencePair, SftExample, Feedback, DatasetVersion, Dataset, Proposal,
                      AgentEvent, TunerMessage, StudioState, Job):  # fmt: skip
            s.exec(delete(model).where(model.project_id == project_id))
        s.delete(project)
        s.commit()
    from slm.inference.engine import engine as infer

    runs_dir = str(get_settings().runs_dir)
    loaded = [str(v or "") for v in (infer.loaded or {}).values()]
    if any(v.startswith(runs_dir) or v in export_paths for v in loaded):
        infer.unload()
    freed = sum(_rmtree(p) for paths in files.values() for p in paths) + sum(_rmtree(p) for p in export_paths)
    tuner.forget(project_id)
    footprint(refresh=True)
    return {"project_id": project_id, "freed_gb": _gb(freed), "kept_exports": keep_exports}
