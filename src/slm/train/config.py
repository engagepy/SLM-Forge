"""Training hyperparameters, hardware-aware presets and trainer YAML generation."""

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from slm import hardware


class TrainConfig(BaseModel):
    mode: Literal["sft", "dpo"] = "sft"
    fine_tune_type: Literal["lora", "dora", "full"] = "lora"

    # Optimisation
    optimizer: Literal["adam", "adamw"] = "adamw"
    weight_decay: float = Field(0.01, ge=0)
    learning_rate: float = Field(1e-4, gt=0)
    lr_schedule: Literal["constant", "cosine", "linear"] = "cosine"
    warmup_steps: int = Field(10, ge=0)
    min_lr_ratio: float = Field(0.1, ge=0, le=1)  # cosine/linear end LR as a fraction of peak

    # Adapter
    lora_rank: int = Field(8, ge=1, le=256)
    lora_scale: float = Field(20.0, gt=0)  # mlx's "scale" plays the role of alpha / rank
    lora_dropout: float = Field(0.0, ge=0, lt=1)
    lora_keys: list[str] | None = None  # e.g. ["self_attn.q_proj", "self_attn.v_proj"]; None = all
    num_layers: int = Field(16, ge=-1)  # -1 = all layers

    # Batching and length
    batch_size: int = Field(4, ge=1)
    grad_accumulation_steps: int = Field(1, ge=1)
    epochs: float | None = Field(2.0, gt=0)
    iters: int | None = Field(None, ge=1)  # overrides epochs when set
    max_seq_length: int = Field(1024, ge=64)
    grad_checkpoint: bool = False
    mask_prompt: bool = True  # train on completions only

    # Bookkeeping
    seed: int = 0
    steps_per_report: int = Field(10, ge=1)
    steps_per_eval: int = Field(50, ge=1)
    save_every: int = Field(100, ge=1)
    val_batches: int = Field(25, ge=-1)

    # Preference optimisation (mode == "dpo")
    beta: float = Field(0.1, gt=0)
    dpo_loss_type: Literal["sigmoid", "hinge", "ipo", "dpop"] = "sigmoid"
    dpop_delta: float = 50.0

    @model_validator(mode="after")
    def _needs_length(self):
        if self.epochs is None and self.iters is None:
            raise ValueError("set epochs or iters")
        return self

    def total_iters(self, n_train: int) -> int:
        """mlx trainers count optimizer iterations; convert epochs to iterations."""
        if self.iters:
            return self.iters
        per_epoch = max(1, math.ceil(n_train / self.batch_size))
        return max(1, math.ceil(per_epoch * (self.epochs or 1)))

    def epochs_for(self, n_train: int) -> float:
        per_epoch = max(1, math.ceil(n_train / self.batch_size))
        return round(self.total_iters(n_train) / per_epoch, 2)

    def effective_warmup(self, iters: int) -> int:
        """Warmup capped at 25% of the run, so short runs (e.g. DPO rounds) reach the peak LR."""
        return min(self.warmup_steps, iters // 4)

    def _schedule(self, iters: int) -> dict | None:
        warmup = self.effective_warmup(iters)
        if self.lr_schedule == "constant" and not warmup:
            return None
        decay_steps = max(1, iters - warmup)
        end = self.learning_rate * self.min_lr_ratio
        if self.lr_schedule == "cosine":
            sched = {"name": "cosine_decay", "arguments": [self.learning_rate, decay_steps, end]}
        elif self.lr_schedule == "linear":
            sched = {"name": "linear_schedule", "arguments": [self.learning_rate, end, decay_steps]}
        else:  # constant after warmup
            sched = {"name": "linear_schedule", "arguments": [self.learning_rate, self.learning_rate, 1]}
        if warmup:
            sched["warmup"] = warmup
        return sched

    def _lora_params(self) -> dict:
        p = {"rank": self.lora_rank, "scale": self.lora_scale, "dropout": self.lora_dropout}
        if self.lora_keys:
            p["keys"] = self.lora_keys
        return p

    def to_trainer_yaml(
        self, *, model: str, data: str, adapter_path: str, n_train: int, resume_adapter_file: str | None = None
    ) -> dict:
        iters = self.total_iters(n_train)
        common = {
            "model": model,
            "train": True,
            "data": data,
            "adapter_path": adapter_path,
            "optimizer": self.optimizer,
            "optimizer_config": {"adamw": {"weight_decay": self.weight_decay}} if self.optimizer == "adamw" else {},
            "learning_rate": self.learning_rate,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "iters": iters,
            "max_seq_length": self.max_seq_length,
            "grad_checkpoint": self.grad_checkpoint,
            "mask_prompt": self.mask_prompt,
            "seed": self.seed,
            "steps_per_report": min(self.steps_per_report, iters),
            "steps_per_eval": min(self.steps_per_eval, iters),
            "save_every": min(self.save_every, iters),
            "val_batches": self.val_batches,
            "lora_parameters": self._lora_params(),
        }
        if sched := self._schedule(iters):
            common["lr_schedule"] = sched
        if resume_adapter_file:
            common["resume_adapter_file"] = resume_adapter_file  # continue an adapter instead of starting one

        if self.mode == "sft":
            return common | {
                "fine_tune_type": self.fine_tune_type,
                "grad_accumulation_steps": self.grad_accumulation_steps,
            }
        return common | {
            "train_mode": "dpo",
            "train_type": self.fine_tune_type,
            "gradient_accumulation_steps": self.grad_accumulation_steps,
            "beta": self.beta,
            "dpo_cpo_loss_type": self.dpo_loss_type,
            "delta": self.dpop_delta,
            "fuse": False,  # we fuse explicitly so lineage stays under our control
        }

    def memory_estimate(self, shape: hardware.ModelShape, typical_len: int | None = None) -> hardware.MemoryEstimate:
        """`typical_len` (e.g. the dataset's p95 token length) replaces the worst case: batches
        are padded to their longest sequence, not to max_seq_length."""
        seq = min(self.max_seq_length, typical_len) if typical_len else self.max_seq_length
        return hardware.estimate_training(
            shape,
            fine_tune_type=self.fine_tune_type,
            batch_size=self.batch_size,
            max_seq_length=seq,
            num_layers=self.num_layers,
            lora_rank=self.lora_rank,
            grad_checkpoint=self.grad_checkpoint,
            preference=self.mode == "dpo",
        )


PresetName = Literal["safe", "balanced", "quality"]


def preset(
    name: PresetName, shape: hardware.ModelShape, mode: str = "sft", typical_len: int | None = None
) -> TrainConfig:
    """Pick settings for this model on this machine, then shrink until the estimate fits.

    `typical_len` is the dataset's p95 token length when known; without it the estimate
    assumes every sequence is max_seq_length long."""
    base: dict = {"mode": mode}
    if name == "safe":
        base |= dict(lora_rank=8, num_layers=8, batch_size=2, max_seq_length=768, grad_checkpoint=True)
    elif name == "balanced":
        base |= dict(lora_rank=16, num_layers=16, batch_size=4, max_seq_length=1024)
    else:  # quality: more capacity, all layers, longer context
        base |= dict(lora_rank=32, lora_scale=16.0, num_layers=-1, batch_size=4, max_seq_length=2048)

    if mode == "dpo":
        # DPO holds a reference model and two sequences per example; start leaner.
        base |= dict(learning_rate=5e-6, epochs=1.0, lr_schedule="constant", warmup_steps=0)
        base["batch_size"] = max(1, base["batch_size"] // 2)

    cfg = TrainConfig(**base)
    # Step down until the estimate fits: checkpointing, then batch (compensate with
    # accumulation), then sequence length.
    for _ in range(12):
        if cfg.memory_estimate(shape, typical_len).fits:
            break
        if not cfg.grad_checkpoint:
            cfg.grad_checkpoint = True
        elif cfg.batch_size > 1:
            cfg.batch_size //= 2
            cfg.grad_accumulation_steps *= 2
        elif cfg.max_seq_length > 256:
            cfg.max_seq_length //= 2
        else:
            break
    return cfg
