"""Job handlers: each turns a Job's config into work and records the resulting artifacts."""

import json
import re
from pathlib import Path

from sqlmodel import Session, select

from slm.config import get_settings
from slm.data import pipeline, scout_tools
from slm.data.clean import CleaningRules
from slm.db import (
    Checkpoint,
    Dataset,
    DatasetVersion,
    PreferencePair,
    Project,
    SftExample,
    engine,
)
from slm.export import fuse as fusing
from slm.models import manage
from slm.train import runner
from slm.train.config import TrainConfig
from slm.train.worker import JobContext, worker


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-")[:60] or "item"


def _run(ctx: JobContext, cmd: list[str], *, parse_metrics: bool = False) -> None:
    def on_line(line: str) -> None:
        ctx.log(line)
        if parse_metrics:
            if (total := runner.parse_total_iters(line)) is not None:
                ctx.result["total_iters"] = total
            if m := runner.parse_line(line):
                ctx.metric(m)
                if m.split == "train":
                    ctx.progress(m.iteration, ctx.result.get("total_iters", 0))

    code = runner.run_process(cmd, log_path=ctx.log_path, on_line=on_line, should_cancel=ctx.should_cancel)
    if code != 0:
        hint = ""
        log = runner.tail(ctx.log_path, 60)
        if "out of memory" in log.lower() or code in (-9, 137):
            hint = " Likely out of memory: try the 'safe' preset, smaller batch or sequence length."
        raise RuntimeError(f"Process exited with code {code}.{hint}")


def _project(s: Session, project_id: int | None) -> Project:
    project = s.get(Project, project_id) if project_id else None
    if project is None:
        raise ValueError("job has no project")
    return project


def _serving_model(project: Project) -> str:
    if project.current_model_path:
        return project.current_model_path
    if project.base_model and (p := manage.local_path_for(project.base_model)):
        return p
    raise ValueError("Project has no downloaded base model yet")


# ── I/O lane ────────────────────────────────────────────────────────────────


@worker.register("download")
def download_job(ctx: JobContext) -> None:
    rec = manage.download(ctx.config["repo_id"], log=ctx.note)
    ctx.result = {"model_id": rec.id, "local_path": rec.local_path, "size_gb": rec.size_gb}
    if ctx.project_id:
        with Session(engine()) as s:
            project = _project(s, ctx.project_id)
            if project.base_model == rec.repo_id and not project.current_model_path:
                project.current_model_path = rec.local_path
                s.add(project)
                s.commit()


@worker.register("import_dataset")
def import_dataset_job(ctx: JobContext) -> None:
    c = ctx.config
    repo = c["repo_id"]
    name = c.get("name") or _slug(repo.split("/")[-1])
    dest = get_settings().datasets_dir / f"p{ctx.project_id}" / "raw" / f"{name}-job{ctx.job_id}"
    ctx.note(f"Streaming up to {c.get('max_rows', 5000)} rows from {repo} ({c.get('split', 'train')}) ...")
    n, columns = scout_tools.import_hf_dataset(
        repo, dest, config=c.get("config"), split=c.get("split", "train"), max_rows=c.get("max_rows", 5000)
    )
    ctx.note(f"Imported {n} rows with columns {columns}")
    with Session(engine()) as s:
        ds = Dataset(
            project_id=ctx.project_id,
            name=name,
            source="hf",
            source_ref=repo,
            license=c.get("license", ""),
            raw_path=str(dest / "raw.jsonl"),
            n_rows=n,
            columns=columns,
        )
        s.add(ds)
        s.commit()
        s.refresh(ds)
        ctx.result = {"dataset_id": ds.id, "rows": n, "columns": columns}


@worker.register("prepare_dataset")
def prepare_dataset_job(ctx: JobContext) -> None:
    c = ctx.config
    with Session(engine()) as s:
        ds = s.get(Dataset, c["dataset_id"])
        project = _project(s, ds.project_id)
        model_path = _serving_model(project) if project.base_model else None
    ctx.note(f"Preparing '{ds.name}' with mapping {json.dumps(c['mapping'])}")
    version = pipeline.prepare(
        ds,
        c["mapping"],
        rules=CleaningRules(**c.get("rules", {})),
        model_path=model_path,
        max_seq_length=c.get("max_seq_length", 1024),
        valid_frac=c.get("valid_frac", 0.05),
        test_frac=c.get("test_frac", 0.05),
        seed=c.get("seed", 0),
        include_feedback=c.get("include_feedback", False),
    )
    ctx.note(f"Cleaning: {version.cleaning_report}")
    ctx.note(f"Splits: train={version.n_train} valid={version.n_valid} test={version.n_test}")
    if version.token_stats:
        ctx.note(f"Tokens: {version.token_stats}")
    ctx.result = {"dataset_version_id": version.id}


# ── GPU lane ────────────────────────────────────────────────────────────────


def _fuse(ctx: JobContext, model_path: str, adapter_path: str, dest: Path) -> Path:
    ctx.note(f"Fusing adapter {adapter_path} into {model_path} ...")
    _run(ctx, fusing.fuse_command(model_path, adapter_path, dest))
    return dest


def _train(ctx: JobContext, cfg: TrainConfig, *, model_path: str, data_dir: str, n_train: int) -> Path:
    adapter_dir = ctx.run_dir / "adapters"
    yaml_cfg = cfg.to_trainer_yaml(model=model_path, data=data_dir, adapter_path=str(adapter_dir), n_train=n_train)
    config_path = runner.write_config(yaml_cfg, ctx.run_dir / "config.yaml")
    iters = yaml_cfg["iters"]
    ctx.result["total_iters"] = iters
    ctx.result["epochs"] = cfg.epochs_for(n_train)
    ctx.note(f"{cfg.mode.upper()} on {n_train} examples: {iters} iterations (~{ctx.result['epochs']} epochs)")
    _run(ctx, runner.trainer_command(cfg.mode, config_path), parse_metrics=True)
    return adapter_dir


def _final_metrics(job_id: int) -> dict:
    from slm.db import Metric

    with Session(engine()) as s:
        rows = s.exec(select(Metric).where(Metric.job_id == job_id).order_by(Metric.iteration)).all()
    out = {}
    if train := [r for r in rows if r.split == "train"]:
        out["train_loss"] = round(train[-1].values["loss"], 4)
        out["peak_mem_gb"] = round(max(r.values.get("peak_mem_gb", 0) for r in train), 2)
        if "accuracy" in train[-1].values:
            out["reward_accuracy"] = round(train[-1].values["accuracy"], 3)
    if val := [r for r in rows if r.split == "val"]:
        out["val_loss_start"] = round(val[0].values["loss"], 4)
        out["val_loss"] = round(val[-1].values["loss"], 4)
    return out


def _warnings(ctx: JobContext) -> list[dict]:
    from slm.db import Metric
    from slm.train.diagnose import diagnose

    with Session(engine()) as s:
        rows = s.exec(select(Metric).where(Metric.job_id == ctx.job_id).order_by(Metric.iteration)).all()
    warnings = diagnose(
        [(r.iteration, r.values["loss"]) for r in rows if r.split == "train"],
        [(r.iteration, r.values["loss"]) for r in rows if r.split == "val"],
    )
    for w in warnings:
        ctx.note(f"⚠ {w.message}")
    return [w.to_dict() for w in warnings]


@worker.register("sft")
def sft_job(ctx: JobContext) -> None:
    cfg = TrainConfig(**(ctx.config.get("train", {}) | {"mode": "sft"}))
    from_base = ctx.config.get("start_from") == "base"
    with Session(engine()) as s:
        project = _project(s, ctx.project_id)
        version = s.get(DatasetVersion, ctx.config["dataset_version_id"])
        if from_base:
            # A fresh run from the untrained model, e.g. to compare hyperparameters fairly.
            model_path = manage.local_path_for(project.base_model or "")
            if not model_path:
                raise ValueError("Base model not downloaded")
            parent, pending_adapter = None, None
            ctx.note("Starting from the base model (ignoring earlier checkpoints).")
        else:
            model_path = _serving_model(project)
            parent = s.exec(
                select(Checkpoint).where(Checkpoint.project_id == project.id).order_by(Checkpoint.id.desc())
            ).first()
            # Continue from the served state: if the latest checkpoint is an un-fused adapter,
            # fuse it first so this SFT round builds on it rather than replacing it.
            pending_adapter = project.current_adapter_path

    if pending_adapter:
        model_path = str(_fuse(ctx, model_path, pending_adapter, ctx.run_dir / "base-fused"))

    if version.mapping.get("format") == "text" and cfg.mask_prompt:
        # Raw text has no prompt/completion split; mlx_lm rejects masking for it.
        cfg.mask_prompt = False
        ctx.note("Text dataset: training on all tokens (prompt masking doesn't apply).")

    adapter_dir = _train(ctx, cfg, model_path=model_path, data_dir=version.path, n_train=version.n_train)
    metrics = _final_metrics(ctx.job_id)
    with Session(engine()) as s:
        project = _project(s, ctx.project_id)
        ckpt = Checkpoint(
            project_id=project.id,
            job_id=ctx.job_id,
            parent_id=parent.id if parent else None,
            kind="sft",
            base_model_path=model_path,
            adapter_path=str(adapter_dir),
            metrics=metrics,
        )
        s.add(ckpt)
        project.current_model_path = model_path
        project.current_adapter_path = str(adapter_dir)
        s.add(project)
        # Feedback-derived SFT examples that went into this dataset are now consumed.
        for ex_id in version.mapping.get("feedback_example_ids", []):
            if ex := s.get(SftExample, ex_id):
                ex.used_in_job_id = ctx.job_id
                s.add(ex)
        s.commit()
        s.refresh(ckpt)
    ctx.result |= {"checkpoint_id": ckpt.id, "metrics": metrics, "warnings": _warnings(ctx)}


@worker.register("dpo")
def dpo_job(ctx: JobContext) -> None:
    cfg = TrainConfig(**(ctx.config.get("train", {}) | {"mode": "dpo"}))
    with Session(engine()) as s:
        project = _project(s, ctx.project_id)
        model_path = _serving_model(project)
        pending_adapter = project.current_adapter_path
        parent = s.exec(
            select(Checkpoint).where(Checkpoint.project_id == project.id).order_by(Checkpoint.id.desc())
        ).first()

    # DPO's frozen reference is loaded from the same path as the policy, so the policy must
    # be a standalone (fused) model: fuse the latest SFT adapter first.
    if pending_adapter:
        model_path = str(_fuse(ctx, model_path, pending_adapter, ctx.run_dir / "policy-fused"))
        if parent and parent.adapter_path == pending_adapter:
            with Session(engine()) as s:
                p = s.get(Checkpoint, parent.id)
                p.fused_path = model_path
                s.add(p)
                s.commit()

    if vid := ctx.config.get("dataset_version_id"):
        with Session(engine()) as s:
            version = s.get(DatasetVersion, vid)
    else:
        version = pipeline.build_preference_version(
            ctx.project_id, model_path=model_path, max_seq_length=cfg.max_seq_length, seed=cfg.seed
        )
        ctx.note(f"Built preference dataset v{version.id}: {version.n_train} train / {version.n_valid} valid pairs")

    adapter_dir = _train(ctx, cfg, model_path=model_path, data_dir=version.path, n_train=version.n_train)
    metrics = _final_metrics(ctx.job_id)
    with Session(engine()) as s:
        project = _project(s, ctx.project_id)
        ckpt = Checkpoint(
            project_id=project.id,
            job_id=ctx.job_id,
            parent_id=parent.id if parent else None,
            kind="dpo",
            base_model_path=model_path,
            adapter_path=str(adapter_dir),
            metrics=metrics,
        )
        s.add(ckpt)
        project.current_model_path = model_path
        project.current_adapter_path = str(adapter_dir)
        s.add(project)
        for pair_id in version.mapping.get("preference_pair_ids", []):
            if pair := s.get(PreferencePair, pair_id):
                pair.used_in_job_id = ctx.job_id
                s.add(pair)
        s.commit()
        s.refresh(ckpt)
    ctx.result |= {
        "checkpoint_id": ckpt.id,
        "metrics": metrics,
        "dataset_version_id": version.id,
        "warnings": _warnings(ctx),
    }


@worker.register("fuse")
def fuse_job(ctx: JobContext) -> None:
    with Session(engine()) as s:
        ckpt = s.get(Checkpoint, ctx.config["checkpoint_id"])
    dest = _fuse(ctx, ckpt.base_model_path, ckpt.adapter_path, ctx.run_dir / "fused")
    with Session(engine()) as s:
        ckpt = s.get(Checkpoint, ctx.config["checkpoint_id"])
        ckpt.fused_path = str(dest)
        s.add(ckpt)
        project = _project(s, ckpt.project_id)
        if project.current_adapter_path == ckpt.adapter_path:
            project.current_model_path, project.current_adapter_path = str(dest), None
            s.add(project)
        s.commit()
    ctx.result = {"fused_path": str(dest)}


def served_ancestry(s: Session, project: Project) -> list[Checkpoint]:
    """The chain of checkpoints behind the served model, oldest first.

    Only this model's ancestors: abandoned runs and fresh-from-base branches are excluded.
    """
    ckpts = s.exec(select(Checkpoint).where(Checkpoint.project_id == project.id)).all()
    served = next(
        (
            c
            for c in reversed(ckpts)
            if (project.current_adapter_path and c.adapter_path == project.current_adapter_path)
            or (c.fused_path and c.fused_path == project.current_model_path and not project.current_adapter_path)
        ),
        None,
    )
    by_id = {c.id: c for c in ckpts}
    chain = []
    while served is not None:
        chain.append(served)
        served = by_id.get(served.parent_id) if served.parent_id else None
    return list(reversed(chain))


@worker.register("export")
def export_job(ctx: JobContext) -> None:
    c = ctx.config
    with Session(engine()) as s:
        project = _project(s, ctx.project_id)
        model_path = _serving_model(project)
        adapter = project.current_adapter_path
        project_info = project.model_dump(include={"name", "goal", "base_model", "system_prompt"})
        lineage = [{"kind": ck.kind, "job_id": ck.job_id, "metrics": ck.metrics} for ck in served_ancestry(s, project)]

    name = _slug(c.get("name") or f"{project_info['name']}-v{len(lineage)}")
    dest = get_settings().exports_dir / name
    staged = Path(model_path)
    if adapter:
        staged = _fuse(ctx, model_path, adapter, ctx.run_dir / "fused")

    bits = c.get("quantize_bits")
    with open(staged / "config.json") as f:
        already_quantized = bool(json.load(f).get("quantization"))
    if bits and not already_quantized:
        ctx.note(f"Quantizing to {bits}-bit ...")
        if dest.exists():
            import shutil

            shutil.rmtree(dest)
        _run(ctx, fusing.quantize_command(staged, dest, bits))
    else:
        if bits and already_quantized:
            ctx.note("Model is already quantized; exporting as-is.")
        fusing.copy_model(staged, dest)

    fusing.write_model_card(dest, name=name, project=project_info, lineage=lineage, sampling=c.get("sampling", {}))
    size = manage.dir_size_gb(dest)
    ctx.note(f"Exported to {dest} ({size:.2f} GB)")
    ctx.result = {"path": str(dest), "size_gb": round(size, 3), "min_ram_gb": fusing.min_mac_memory_gb(dest)}
