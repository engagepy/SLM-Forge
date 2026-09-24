"""Search the Hugging Face Hub for base models that fit this machine."""

import re
from dataclasses import asdict, dataclass

from huggingface_hub import HfApi

from slm import hardware

_BITS_IN_NAME = re.compile(r"(?:^|[-_])(\d)[-_]?bit(?:$|[-_])", re.IGNORECASE)


@dataclass
class ModelCandidate:
    id: str
    downloads: int
    likes: int
    params: int
    bits: float
    weights_gb: float
    train_estimate_gb: float
    fit: str  # fits | tight | too_big | unknown
    gated: bool
    is_mlx: bool
    architecture: str
    last_modified: str

    def to_dict(self) -> dict:
        return asdict(self)


def infer_bits(repo_id: str, config: dict | None, dtypes: dict | None) -> float:
    cfg = config or {}
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    if "bits" in q:
        return float(q["bits"])
    if m := _BITS_IN_NAME.search(repo_id.split("/")[-1]):
        return float(m.group(1))
    if dtypes and any(k.upper() in ("U32", "U8", "I8") for k in dtypes):
        return 4.0  # packed integer weights without explicit metadata; assume 4-bit
    return 16.0


def fit_verdict(train_gb: float, budget_gb: float) -> str:
    if train_gb <= budget_gb * 0.75:
        return "fits"
    if train_gb <= budget_gb:
        return "tight"
    return "too_big"


def rough_training_gb(params: int, bits: float) -> float:
    """Quick LoRA estimate from parameter count alone (no config needed).

    Weights plus ~35% for LoRA state and optimizer, plus ~1.5 GB of activations/logits at
    batch 4 × 1024 tokens. `hardware.estimate_training` refines this once config.json is local.
    """
    weights = params * (bits + (0.5 if bits < 16 else 0)) / 8 / hardware.GB
    return weights * 1.35 + 1.5 + 0.6


def search_models(
    query: str = "",
    *,
    mlx_only: bool = True,
    max_params_b: float | None = None,
    include_too_big: bool = False,
    limit: int = 40,
) -> list[ModelCandidate]:
    budget = hardware.detect().budget_gb
    api = HfApi()
    results = api.list_models(
        author="mlx-community" if mlx_only else None,
        search=query or None,
        pipeline_tag="text-generation",
        sort="downloads",
        limit=limit * 3,  # over-fetch: some rows lack size info or get filtered
        expand=["downloads", "likes", "safetensors", "config", "gated", "lastModified", "tags"],
    )
    out: list[ModelCandidate] = []
    for m in results:
        st = m.safetensors
        if not st or not st.total:
            continue
        params = int(st.total)
        if max_params_b and params > max_params_b * 1e9:
            continue
        bits = infer_bits(m.id, m.config, st.parameters)
        weights = params * (bits + (0.5 if bits < 16 else 0)) / 8 / hardware.GB
        train_gb = rough_training_gb(params, bits)
        fit = fit_verdict(train_gb, budget)
        if fit == "too_big" and not include_too_big:
            continue
        arch = ((m.config or {}).get("architectures") or [""])[0]
        out.append(
            ModelCandidate(
                id=m.id,
                downloads=m.downloads or 0,
                likes=m.likes or 0,
                params=params,
                bits=bits,
                weights_gb=round(weights, 2),
                train_estimate_gb=round(train_gb, 2),
                fit=fit,
                gated=bool(m.gated),
                is_mlx=m.id.startswith("mlx-community/") or "mlx" in (m.tags or []),
                architecture=arch,
                last_modified=str(m.last_modified or ""),
            )
        )
        if len(out) >= limit:
            break
    return out


def fetch_config(repo_id: str) -> dict:
    """Download just config.json (a few KB) to get the exact model shape."""
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "config.json")
    with open(path) as f:
        return json.load(f)
