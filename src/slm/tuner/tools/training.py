"""Planning and proposing runs: training, roll-back, export."""

from sqlmodel import Session, select

from slm.db import (
    Checkpoint,
    DatasetVersion,
    Job,
    Metric,
    Project,
    engine,
)
from slm.events import canvas_changed
from slm.inference.engine import engine as infer
from slm.models import manage
from slm.sessions import overview
from slm.train.config import TrainConfig, preset
from slm.train.worker import worker
from slm.tuner import confirm
from slm.tuner.tools._core import Ctx, _job_brief, tool


def _estimated_minutes(pid: int, iterations: int, params_b: float, seq_len: int, batch: int) -> dict:
    """Minutes a run should take: from this Mac's last run of similar size when there is one,
    else a rule of thumb for 4-bit LoRA on Apple Silicon (calibrated at 512 tokens × batch 4)."""
    with Session(engine()) as s:
        last = s.exec(
            select(Metric)
            .join(Job, Job.id == Metric.job_id)
            .where(Job.project_id == pid, Job.kind.in_(["sft", "dpo"]), Metric.split == "train")
            .order_by(Metric.id.desc())
            .limit(20)
        ).all()
    rates = [m.values.get("it_per_sec") for m in last if m.values.get("it_per_sec")]
    if rates:
        rate, basis = sorted(rates)[len(rates) // 2], "this project's last run"
    else:
        rate = 1.3 if params_b <= 0.7 else 0.8 if params_b <= 1.6 else 0.45 if params_b <= 3.5 else 0.2
        rate *= (512 * 4) / max(1, seq_len * batch)  # rough: time scales with tokens per step
        basis = "rule of thumb for this size"
    return {"minutes": max(1, round(iterations / rate / 60)), "basis": basis}


def _plan(
    pid: int, mode: str, preset_name: str, dataset_version_id: int | None, **overrides
) -> tuple[TrainConfig, dict, DatasetVersion | None]:
    """The training config for these settings, and what the Tuner needs to know about it."""
    with Session(engine()) as s:
        project = s.get(Project, pid)
        version = s.get(DatasetVersion, dataset_version_id) if dataset_version_id else None
        if version is not None and version.project_id != pid:
            raise ValueError("no such dataset version in this project")
    if mode == "sft" and version is None:
        raise ValueError("SFT needs dataset_version_id (prepare a dataset first)")
    path = manage.serving_path(project)
    if not path:
        raise ValueError("Download a base model first")
    shape = manage.read_shape(path)
    typical = (version.token_stats or {}).get("p95") if version else None
    cfg = preset(preset_name, shape, mode, typical)
    cfg = TrainConfig(**(cfg.model_dump() | {k: v for k, v in overrides.items() if v is not None}))
    est = cfg.memory_estimate(shape, typical)
    info = {
        "memory_gb": round(est.total_gb, 2),
        "budget_gb": round(est.budget_gb, 1),
        "fits": est.fits,
        "learning_rate": cfg.learning_rate,
        "lora_rank": cfg.lora_rank,
        "layers_tuned": cfg.num_layers,
        "batch_size": cfg.batch_size,
        "grad_accumulation": cfg.grad_accumulation_steps,
        "max_seq_length": cfg.max_seq_length,
        "grad_checkpoint": cfg.grad_checkpoint,
    }
    if version:
        iters = cfg.total_iters(version.n_train)
        info |= {
            "iterations": iters,
            "epochs": cfg.epochs_for(version.n_train),
            "train_examples": version.n_train,
            "estimated": _estimated_minutes(pid, iters, shape.params / 1e9, cfg.max_seq_length, cfg.batch_size),
        }
    return cfg, info, version


@tool
def plan_training(
    ctx: Ctx,
    mode: str = "sft",
    dataset_version_id: int | None = None,
    preset_name: str = "balanced",
    learning_rate: float | None = None,
    epochs: float | None = None,
    iters: int | None = None,
    lora_rank: int | None = None,
    max_seq_length: int | None = None,
    batch_size: int | None = None,
) -> dict:
    """Work out a training configuration without starting it: memory needed vs. this Mac's budget,
    iterations and epochs, and the key settings. mode: sft | dpo. preset_name: safe | balanced | quality.
    Guidance: for SFT, learning_rate 1e-4 is a good default (2e-4 has diverged on these models);
    1–3 epochs; max_seq_length should cover the data's p95 tokens. For DPO use lr ~5e-6, 1–2 epochs."""
    _, info, _ = _plan(
        ctx.context.project_id, mode, preset_name, dataset_version_id,
        learning_rate=learning_rate, epochs=epochs, iters=iters, lora_rank=lora_rank,
        max_seq_length=max_seq_length, batch_size=batch_size,
    )  # fmt: skip
    return info


@tool
def start_training(
    ctx: Ctx,
    mode: str = "sft",
    dataset_version_id: int | None = None,
    preset_name: str = "balanced",
    learning_rate: float | None = None,
    epochs: float | None = None,
    iters: int | None = None,
    lora_rank: int | None = None,
    max_seq_length: int | None = None,
    batch_size: int | None = None,
    start_from: str = "current",
    reason: str = "",
) -> dict:
    """Propose a training run. It starts only when the user confirms it on a card; you'll be told
    when they do, and again when it finishes (progress shows live on the canvas). mode sft needs
    dataset_version_id; mode dpo trains on the feedback pairs (needs at least 3). start_from:
    current (continue from the served model) | base (fresh run). reason: one or two plain
    sentences, shown on the card, on what this run should teach the model and roughly how long it takes."""
    pid = ctx.context.project_id
    if mode not in ("sft", "dpo"):
        raise ValueError("mode must be sft or dpo")
    cfg, info, version = _plan(
        pid, mode, preset_name, dataset_version_id,
        learning_rate=learning_rate, epochs=epochs, iters=iters, lora_rank=lora_rank,
        max_seq_length=max_seq_length, batch_size=batch_size,
    )  # fmt: skip
    if not info["fits"]:
        return {"status": "won't fit", **info, "hint": "use preset safe, lower batch_size or max_seq_length"}
    config = {"train": cfg.model_dump(), "start_from": start_from}
    if version:
        config["dataset_version_id"] = version.id
    ahead = overview()["capacity"]["gpu_running"]
    if ahead:
        info["would_queue_behind"] = ahead
    title = (
        f"Train on {version.n_train} examples ({info['iterations']} steps)"
        if mode == "sft" and version
        else "Refine with a preference round (DPO)"
    )
    return confirm.propose(pid, mode, title, reason, info, {"config": config}) | info


@tool
def training_progress(ctx: Ctx, job_id: int) -> dict:
    """Current state of a job: status, iteration, latest train/validation loss, memory and warnings."""
    with Session(engine()) as s:
        job = s.get(Job, job_id)
        if job is None or job.project_id != ctx.context.project_id:
            raise ValueError("no such job in this project")
        metrics = s.exec(select(Metric).where(Metric.job_id == job_id).order_by(Metric.iteration)).all()
    train = [m for m in metrics if m.split == "train"]
    val = [m for m in metrics if m.split == "val"]
    out = _job_brief(job)
    if train:
        out["latest_train"] = {
            "iteration": train[-1].iteration,
            **{k: round(v, 4) for k, v in train[-1].values.items()},
        }
        out["first_train_loss"] = round(train[0].values["loss"], 4)
    if val:
        out["val_loss_curve"] = [round(m.values["loss"], 4) for m in val]
    return out


@tool
def cancel_job(ctx: Ctx, job_id: int) -> str:
    """Stop a queued or running job."""
    return worker.cancel(job_id)


@tool
def serve_checkpoint(ctx: Ctx, checkpoint_id: int) -> dict:
    """Make the project serve an earlier checkpoint again (a roll-back). Use it when a round made
    the model worse (a val_worse or overfitting warning, or a lower evaluate_model score): later
    runs continue from the served checkpoint, and export packages it. get_status lists checkpoints."""
    pid = ctx.context.project_id
    from slm.train.jobs import model_in_use
    from slm.train.jobs import serve_checkpoint as _serve

    with Session(engine()) as s:
        if (busy := model_in_use(s, pid)) is not None:
            raise ValueError(f"Job {busy} is using this project's model; wait for it to finish.")
        project = s.get(Project, pid)
        c = s.get(Checkpoint, checkpoint_id)
        if c is None or c.project_id != pid:
            raise ValueError("no such checkpoint in this project")
        _serve(s, project, c)
        out = {"serving": f"checkpoint:{c.id}", "kind": c.kind, "job_id": c.job_id, "metrics": c.metrics}
    infer.unload()  # the next generation loads the rolled-back weights
    canvas_changed(pid)
    return out


@tool
def export_model(ctx: Ctx, name: str, quantize_bits: int | None = None, reason: str = "") -> dict:
    """Propose packaging the current model: fuse adapters, optionally quantize (4/6/8; skip if the
    base is already quantized), and write a model card. It runs only once the user confirms it on
    a card. reason: one plain sentence for the card. Afterwards they can chat with the exported
    model on the "Try it" page."""
    from slm.export.fuse import check_bits

    check_bits(quantize_bits)
    config = {"name": name, "quantize_bits": quantize_bits, "sampling": {"temperature": 0.7, "top_p": 0.95}}
    details = {"name": name, "quantize_bits": quantize_bits}
    return confirm.propose(ctx.context.project_id, "export", f"Export the model as “{name}”", reason, details,
                           {"config": config})  # fmt: skip
