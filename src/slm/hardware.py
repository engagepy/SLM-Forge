"""Hardware detection and training/inference memory estimates.

Estimates are deliberately conservative heuristics, not exact accounting. The training
runner records real peak memory from MLX so estimates can be checked against reality.
"""

import math
import platform
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Literal

GB = 1024**3
# Headroom left for macOS, the API server and the browser while training.
SYSTEM_HEADROOM_GB = 1.5


@dataclass(frozen=True)
class HardwareInfo:
    chip: str
    total_memory_gb: float
    gpu_working_set_gb: float
    cpu_cores: int
    macos: str

    @property
    def budget_gb(self) -> float:
        """Memory a training or inference job can safely use."""
        return max(self.gpu_working_set_gb - SYSTEM_HEADROOM_GB, 1.0)

    def to_dict(self) -> dict:
        return {**asdict(self), "budget_gb": round(self.budget_gb, 2)}


def _sysctl(key: str) -> str:
    return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()


@lru_cache
def detect() -> HardwareInfo:
    import mlx.core as mx

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return HardwareInfo(
        chip=str(info.get("device_name") or _sysctl("machdep.cpu.brand_string")),
        total_memory_gb=round(int(info["memory_size"]) / GB, 2),
        gpu_working_set_gb=round(int(info["max_recommended_working_set_size"]) / GB, 2),
        cpu_cores=int(_sysctl("hw.ncpu")),
        macos=platform.mac_ver()[0],
    )


@dataclass(frozen=True)
class ModelShape:
    """The parts of a model's config.json that drive memory use."""

    hidden_size: int
    num_layers: int
    intermediate_size: int
    vocab_size: int
    num_heads: int
    num_kv_heads: int
    tie_embeddings: bool = True
    bits: float = 16.0  # 4 or 8 for quantized checkpoints

    @classmethod
    def from_config(cls, cfg: dict) -> "ModelShape":
        text = cfg.get("text_config", cfg)  # multimodal configs nest the LM
        heads = text.get("num_attention_heads", 1)
        q = cfg.get("quantization") or cfg.get("quantization_config") or {}
        return cls(
            hidden_size=text["hidden_size"],
            num_layers=text["num_hidden_layers"],
            intermediate_size=text.get("intermediate_size", 4 * text["hidden_size"]),
            vocab_size=text["vocab_size"],
            num_heads=heads,
            num_kv_heads=text.get("num_key_value_heads", heads),
            tie_embeddings=cfg.get("tie_word_embeddings", True),
            bits=float(q.get("bits", 16)),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def params(self) -> int:
        h, i = self.hidden_size, self.intermediate_size
        kv = self.num_kv_heads * self.head_dim
        attn = 2 * h * h + 2 * h * kv  # q, o + k, v
        mlp = 3 * h * i  # gate, up, down
        embed = self.vocab_size * h * (1 if self.tie_embeddings else 2)
        return self.num_layers * (attn + mlp) + embed


def weights_gb(shape: ModelShape) -> float:
    # Quantized formats also store a scale and bias per group of 64 (≈0.5 extra bits).
    effective_bits = shape.bits + (0.5 if shape.bits < 16 else 0)
    return shape.params * effective_bits / 8 / GB


FineTuneType = Literal["lora", "dora", "full"]


@dataclass
class MemoryEstimate:
    weights_gb: float
    trainable_gb: float
    activations_gb: float
    logits_gb: float
    overhead_gb: float
    total_gb: float
    budget_gb: float
    fits: bool
    headroom_gb: float

    def to_dict(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}


def estimate_training(
    shape: ModelShape,
    *,
    fine_tune_type: FineTuneType = "lora",
    batch_size: int = 4,
    max_seq_length: int = 1024,
    num_layers: int = 16,
    lora_rank: int = 8,
    grad_checkpoint: bool = False,
    preference: bool = False,
    budget_gb: float | None = None,
) -> MemoryEstimate:
    """Estimate peak memory for an SFT or preference (DPO) training run."""
    budget = budget_gb if budget_gb is not None else detect().budget_gb
    tuned_layers = shape.num_layers if num_layers < 0 else min(num_layers, shape.num_layers)
    w = weights_gb(shape)

    if fine_tune_type == "full":
        # bf16 weights + bf16 grads + two fp32 Adam moments; quantized weights can't be trained.
        trainable = shape.params * (2 + 8) / GB
    else:
        # LoRA A and B on the 7 linear projections per tuned layer, fp32 + two Adam moments.
        h, i = shape.hidden_size, shape.intermediate_size
        per_layer = lora_rank * (4 * (h + h) + 3 * (h + i))
        trainable = tuned_layers * per_layer * 4 * 3 / GB

    # Sequences in flight: DPO runs chosen and rejected through the policy.
    seqs = batch_size * (2 if preference else 1)
    tokens = seqs * max_seq_length
    # Per token per tuned layer, keep ~16 hidden-sized bf16 tensors for backprop;
    # checkpointing drops that to roughly the layer input.
    per_token_layer = shape.hidden_size * 2 * (2 if grad_checkpoint else 16)
    # MLP intermediate activations dominate when not checkpointed.
    if not grad_checkpoint:
        per_token_layer += shape.intermediate_size * 2 * 3
    activations = tokens * tuned_layers * per_token_layer / GB
    logits = tokens * shape.vocab_size * 4 / GB  # fp32 logits for the loss

    ref_weights = w if preference else 0.0  # DPO keeps a frozen reference copy
    overhead = 0.6
    total = w + ref_weights + trainable + activations + logits + overhead
    return MemoryEstimate(
        weights_gb=w + ref_weights,
        trainable_gb=trainable,
        activations_gb=activations,
        logits_gb=logits,
        overhead_gb=overhead,
        total_gb=total,
        budget_gb=budget,
        fits=total <= budget,
        headroom_gb=budget - total,
    )


def estimate_inference(shape: ModelShape, context_tokens: int = 4096, budget_gb: float | None = None) -> MemoryEstimate:
    budget = budget_gb if budget_gb is not None else detect().budget_gb
    w = weights_gb(shape)
    kv_cache = 2 * shape.num_layers * shape.num_kv_heads * shape.head_dim * context_tokens * 2 / GB
    overhead = 0.4
    total = w + kv_cache + overhead
    return MemoryEstimate(
        weights_gb=w,
        trainable_gb=0.0,
        activations_gb=kv_cache,
        logits_gb=0.0,
        overhead_gb=overhead,
        total_gb=total,
        budget_gb=budget,
        fits=total <= budget,
        headroom_gb=budget - total,
    )


def max_params_for_budget(budget_gb: float, bits: float = 4.0) -> int:
    """Largest parameter count whose weights plus typical LoRA training fit the budget.

    Rule of thumb: LoRA training peaks around 2.2x the quantized weight size plus ~2 GB for
    activations and logits at modest batch/sequence settings (larger models have wider layers,
    so activations grow with them). On a 16 GB Mac this puts ~7B at 4-bit right at the edge.
    """
    usable = max(budget_gb - 2.0, 0.5)
    bytes_per_param = (bits + 0.5) / 8
    return math.floor(usable / 2.2 * GB / bytes_per_param)
