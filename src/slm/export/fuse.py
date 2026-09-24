"""Fuse adapters into standalone models and package exports with a model card."""

import json
import shutil
import sys
from datetime import date
from pathlib import Path

from slm import hardware


def fuse_command(model_path: str, adapter_path: str, save_path: Path, dequantize: bool = False) -> list[str]:
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_path,
        "--adapter-path", adapter_path,
        "--save-path", str(save_path),
    ]  # fmt: skip
    if dequantize:
        cmd.append("--dequantize")
    return cmd


def quantize_command(src: Path, dest: Path, bits: int, group_size: int = 64) -> list[str]:
    return [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", str(src),
        "--mlx-path", str(dest),
        "-q", "--q-bits", str(bits), "--q-group-size", str(group_size),
    ]  # fmt: skip


def min_mac_memory_gb(model_dir: Path, context_tokens: int = 4096) -> int:
    """Smallest standard Mac memory tier that runs this model comfortably."""
    with open(model_dir / "config.json") as f:
        shape = hardware.ModelShape.from_config(json.load(f))
    need = hardware.estimate_inference(shape, context_tokens, budget_gb=1e9).total_gb
    # macOS lets the GPU use roughly 65-75% of RAM; size the tier so `need` fits in ~65%.
    for tier in (8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192):
        if need <= tier * 0.65:
            return tier
    return 256


def write_model_card(dest: Path, *, name: str, project: dict, lineage: list[dict], sampling: dict) -> None:
    ram = min_mac_memory_gb(dest)
    lines = [
        f"# {name}",
        "",
        f"Fine-tuned locally with SLM Forge on {date.today().isoformat()}.",
        "",
        f"**Goal:** {project.get('goal') or '—'}",
        f"**Base model:** `{project.get('base_model')}`",
        f"**Runs on:** Apple Silicon Macs with **{ram} GB** unified memory or more.",
        "",
        "## Lineage",
        "",
        *[
            f"{i + 1}. **{step['kind'].upper()}** (job {step.get('job_id')}): "
            + ", ".join(f"{k} {v}" for k, v in step.get("metrics", {}).items())
            for i, step in enumerate(lineage)
        ],
        "",
        "## Usage",
        "",
        "```bash",
        "pip install mlx-lm",
        f'mlx_lm.generate --model {dest} --prompt "Hello"',
        "```",
        "",
        "## Recommended sampling",
        "",
        "```json",
        json.dumps(sampling, indent=2),
        "```",
    ]
    if project.get("system_prompt"):
        lines += ["", "## System prompt", "", "```", project["system_prompt"], "```"]
    (dest / "README.md").write_text("\n".join(lines) + "\n")
    (dest / "slm_forge.json").write_text(
        json.dumps(
            {"project": project, "lineage": lineage, "sampling": sampling, "min_ram_gb": ram}, indent=2, default=str
        )
    )


def copy_model(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
