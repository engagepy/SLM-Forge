"""Job handlers: each turns a Job's config into work and records the resulting artifacts."""

import json
import re
import shutil
from pathlib import Path

from sqlmodel import Session, select

from slm.agents.base import set_proposal_status
from slm.config import get_settings
from slm.data import pipeline, scout_tools
from slm.data.clean import CleaningRules
from slm.db import (
    Checkpoint,
    Dataset,
    DatasetVersion,
    Job,
    Metric,
    PreferencePair,
    Project,
    Proposal,
    SftExample,
    engine,
)
from slm.export import fuse as fusing
from slm.models import manage
from slm.train import runner
from slm.train.config import TrainConfig
from slm.train.diagnose import Warning, diagnose
from slm.train.worker import JobContext, worker


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s).strip("-")[:60] or "item"


def _run(ctx: JobContext, cmd: list[str], *, parse_metrics: bool = False) -> None:
    def on_line(line: str) -> None:
        ctx.log(line)
        # The trainer warns (once per batch, collapsed by the runner) when it truncates examples.
        if n := runner.truncation_count(line):
            ctx.result["truncated_batches"] = ctx.result.get("truncated_batches", 0) + n
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
    if path := manage.serving_path(project):
        return path
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
    max_rows = int(c.get("max_rows", 5000))
    ctx.note(f"Streaming up to {max_rows:,} rows from {repo} ({c.get('split', 'train')}) ...")

    def progress(n: int) -> None:
        ctx.progress(n, max_rows)
        if n % 10000 == 0:
            ctx.note(f"{n:,} rows so far")

    n, columns = scout_tools.import_hf_dataset(
        repo, dest, config=c.get("config"), split=c.get("split", "train"), max_rows=max_rows, on_progress=progress
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
        max_examples=c.get("max_examples"),
        valid_frac=c.get("valid_frac", 0.05),
        test_frac=c.get("test_frac", 0.05),
        seed=c.get("seed", 0),
        include_feedback=c.get("include_feedback", False),
    )
    ctx.note(f"Cleaning: {version.cleaning_report}")
    ctx.note(f"Splits: train={version.n_train} valid={version.n_valid} test={version.n_test}")
    if version.token_stats:
        ctx.note(f"Tokens: {version.token_stats}")
    ctx.result = {
        "dataset_version_id": version.id,
        "n_train": version.n_train,
        "kept": version.cleaning_report.get("kept"),
        "input_rows": version.cleaning_report.get("input_rows"),
    }
    # A pending DataPrep suggestion for this dataset has now been acted on; close it so the
    # UI doesn't keep offering it. (Approving it directly already marks it executed.)

    with Session(engine()) as s:
        stale = s.exec(
            select(Proposal).where(
                Proposal.project_id == ds.project_id,
                Proposal.action == "prepare_dataset",
                Proposal.status == "pending",
            )
        ).all()
        stale_ids = [p.id for p in stale if p.payload.get("dataset_id") == ds.id]
    for pid in stale_ids:
        set_proposal_status(pid, "executed", {"dataset_version_id": version.id, "via": "manual prepare"})


# ── GPU lane ────────────────────────────────────────────────────────────────


def _fuse(ctx: JobContext, model_path: str, adapter_path: str, dest: Path) -> Path:
    ctx.note(f"Fusing adapter {adapter_path} into {model_path} ...")
    _run(ctx, fusing.fuse_command(model_path, adapter_path, dest))
    return dest


def match_adapter(cfg: TrainConfig, adapter_dir: Path, note=lambda _: None) -> TrainConfig:
    """Resuming an adapter needs the same LoRA shape it was trained with: take rank, scale and
    layer count from its adapter_config.json, whatever the preset says."""
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return cfg
    saved = json.loads(cfg_path.read_text())
    lora = saved.get("lora_parameters") or {}
    changes = {}
    if saved.get("num_layers") and saved["num_layers"] != cfg.num_layers:
        changes["num_layers"] = saved["num_layers"]
    if lora.get("rank") and lora["rank"] != cfg.lora_rank:
        changes["lora_rank"] = lora["rank"]
    if lora.get("scale") and lora["scale"] != cfg.lora_scale:
        changes["lora_scale"] = lora["scale"]
    if changes:
        note(f"Continuing the served adapter, so its LoRA shape applies: {changes}")
        cfg = TrainConfig(**(cfg.model_dump() | changes))
    return cfg


def _train(
    ctx: JobContext, cfg: TrainConfig, *, model_path: str, data_dir: str, n_train: int, resume_from: str | None = None
) -> Path:
    adapter_dir = ctx.run_dir / "adapters"
    yaml_cfg = cfg.to_trainer_yaml(
        model=model_path, data=data_dir, adapter_path=str(adapter_dir), n_train=n_train, resume_adapter_file=resume_from
    )
    config_path = runner.write_config(yaml_cfg, ctx.run_dir / "config.yaml")
    iters = yaml_cfg["iters"]
    ctx.result["total_iters"] = iters
    ctx.result["epochs"] = cfg.epochs_for(n_train)
    ctx.note(f"{cfg.mode.upper()} on {n_train} examples: {iters} iterations (~{ctx.result['epochs']} epochs)")
    _run(ctx, runner.trainer_command(cfg.mode, config_path), parse_metrics=True)
    return adapter_dir


def _final_metrics(job_id: int) -> dict:

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

    with Session(engine()) as s:
        rows = s.exec(select(Metric).where(Metric.job_id == ctx.job_id).order_by(Metric.iteration)).all()
    warnings = diagnose(
        [(r.iteration, r.values["loss"]) for r in rows if r.split == "train"],
        [(r.iteration, r.values["loss"]) for r in rows if r.split == "val"],
    )
    if ctx.result.get("total_iters") and not any(r.split == "train" for r in rows):
        warnings.append(
            Warning(
                "no_metrics",
                "The trainer ran but none of its progress lines were understood (its log format may have "
                "changed), so there are no loss curves and no diagnostics for this run. Treat its result as "
                "unverified and check the job log.",
            )
        )
    if n := ctx.result.get("truncated_batches"):
        warnings.append(
            Warning(
                "truncated",
                f"The trainer cut examples short in {n} batch(es): some were longer than max_seq_length, so the "
                "model learned answers that stop mid-way. Rebuild the dataset (the served model's tokenizer gives "
                "exact lengths) or lower max_seq_length, and don't trust this run's loss.",
            )
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
            parent = served_checkpoint(s, project)  # the served one, not the newest: roll-backs matter
            # Continue from the served state. An un-fused adapter is continued in place (the trainer
            # resumes its weights on the same base), rather than fused into a 1–2 GB copy per run.
            pending_adapter = project.current_adapter_path

    resume = None
    if pending_adapter:
        weights = Path(pending_adapter) / "adapters.safetensors"
        if weights.exists():
            resume = str(weights)
            cfg = match_adapter(cfg, Path(pending_adapter), ctx.note)
            ctx.note(f"Continuing adapter {pending_adapter} on {model_path}")
        else:  # fusing needs the same file, so there is nothing to continue from
            raise ValueError(
                f"The served adapter has no weights on disk ({weights}). Roll back to a checkpoint that "
                "has them, or start from the base model."
            )

    if version.mapping.get("format") == "text" and cfg.mask_prompt:
        # Raw text has no prompt/completion split; mlx_lm rejects masking for it.
        cfg.mask_prompt = False
        ctx.note("Text dataset: training on all tokens (prompt masking doesn't apply).")

    adapter_dir = _train(
        ctx, cfg, model_path=model_path, data_dir=version.path, n_train=version.n_train, resume_from=resume
    )
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
        parent = served_checkpoint(s, project)

    # DPO's frozen reference is loaded from the same path as the policy, so the policy must be a
    # standalone (fused) model: fuse the latest SFT adapter first. This is the one place a fused
    # copy is still written per run; the Storage page reclaims it once nothing builds on it.
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


def serve_checkpoint(s: Session, project: Project, ckpt: Checkpoint) -> None:
    """Make the project serve this checkpoint (a roll-back, or forward). Later runs continue from it."""
    if ckpt.fused_path:
        project.current_model_path, project.current_adapter_path = ckpt.fused_path, None
    else:
        project.current_model_path, project.current_adapter_path = ckpt.base_model_path, ckpt.adapter_path
    s.add(project)
    s.commit()


def served_checkpoint(s: Session, project: Project) -> Checkpoint | None:
    """The checkpoint the project serves right now (None: the plain base). After a roll-back this
    is not the newest checkpoint, and it is the parent every new run must record."""
    ckpts = s.exec(select(Checkpoint).where(Checkpoint.project_id == project.id).order_by(Checkpoint.id)).all()
    return next(
        (
            c
            for c in reversed(ckpts)
            if (project.current_adapter_path and c.adapter_path == project.current_adapter_path)
            or (c.fused_path and c.fused_path == project.current_model_path and not project.current_adapter_path)
        ),
        None,
    )


def served_ancestry(s: Session, project: Project) -> list[Checkpoint]:
    """The chain of checkpoints behind the served model, oldest first.

    Only this model's ancestors: abandoned runs and fresh-from-base branches are excluded.
    """
    ckpts = s.exec(select(Checkpoint).where(Checkpoint.project_id == project.id)).all()
    served = served_checkpoint(s, project)
    by_id = {c.id: c for c in ckpts}
    chain = []
    while served is not None:
        chain.append(served)
        served = by_id.get(served.parent_id) if served.parent_id else None
    return list(reversed(chain))


def model_in_use(s: Session, project_id: int) -> int | None:
    """The id of a queued or running job that trains or packages this project's model, if any:
    switching the served checkpoint under it would corrupt its lineage."""
    return s.exec(
        select(Job.id).where(
            Job.project_id == project_id,
            Job.kind.in_(["sft", "dpo", "export"]),
            Job.status.in_(["queued", "running"]),
        )
    ).first()


@worker.register("export")
def export_job(ctx: JobContext) -> None:
    c = ctx.config
    with Session(engine()) as s:
        project = _project(s, ctx.project_id)
        model_path = _serving_model(project)
        adapter = project.current_adapter_path
        project_info = project.model_dump(include={"name", "goal", "base_model", "system_prompt"})
        lineage = [{"kind": ck.kind, "job_id": ck.job_id, "metrics": ck.metrics} for ck in served_ancestry(s, project)]

    # A new folder every time: re-exporting under a used name must never replace an earlier model.
    dest = fusing.unique_dest(
        get_settings().exports_dir, _slug(c.get("name") or f"{project_info['name']}-v{len(lineage)}")
    )
    name = dest.name
    staged = Path(model_path)
    if adapter:
        staged = _fuse(ctx, model_path, adapter, ctx.run_dir / "fused")

    bits = c.get("quantize_bits")
    with open(staged / "config.json") as f:
        already_quantized = bool(json.load(f).get("quantization"))
    if bits and not already_quantized:
        ctx.note(f"Quantizing to {bits}-bit ...")
        _run(ctx, fusing.quantize_command(staged, dest, bits))
    else:
        if bits and already_quantized:
            ctx.note("Model is already quantized; exporting as-is.")
        shutil.copytree(staged, dest)

    built_in = fusing.bake_system_prompt(dest, project_info.get("system_prompt") or "")
    if built_in:
        ctx.note("Built the system prompt into the chat template: the export answers like the app with no flags.")
    elif project_info.get("system_prompt"):
        ctx.note("Could not build the system prompt into this chat template: pass --system-prompt when running it.")
    fusing.write_model_card(
        dest, name=name, project=project_info, lineage=lineage, sampling=c.get("sampling", {}), built_in=built_in
    )
    size = manage.dir_size_gb(dest)
    ctx.note(f"Exported to {dest} ({size:.2f} GB)")
    ctx.result = {
        "path": str(dest),
        "size_gb": round(size, 3),
        "min_ram_gb": fusing.min_mac_memory_gb(dest),
        "system_prompt_built_in": built_in,
    }
