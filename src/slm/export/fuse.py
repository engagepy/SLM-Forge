"""Fuse adapters into standalone models and package exports with a model card."""

import json
import re
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


# The chat template's default system message: what the model is told when the caller gives no
# system prompt. Qwen-style templates spell it out as a literal in an else-branch after checking
# for a system message, sometimes wrapped in the role markup ('<|im_start|>system\n...<|im_end|>\n').
_DEFAULT_SYSTEM = re.compile(r"(\[0\]\['role'\] == 'system' %\}.*?\{%-? else -?%\}\s*\{\{-? ')([^']*)(' -?\}\})", re.S)


def _jinja_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def _swap_default(template: str, system_prompt: str) -> tuple[str, int]:
    """Replace the default system sentence in every else-branch, keeping any role markup around it."""
    literal = _jinja_literal(system_prompt)

    def swap(m: re.Match) -> str:
        body = m.group(2)
        pre = post = ""
        if "system\\n" in body:  # the literal carries the markup: keep it
            cut = body.index("system\\n") + len("system\\n")
            pre, body = body[:cut], body[cut:]
        if "<|im_end|>" in body:
            cut = body.index("<|im_end|>")
            body, post = body[:cut], body[cut:]
        return m.group(1) + pre + literal + post + m.group(3)

    return _DEFAULT_SYSTEM.subn(swap, template)


def bake_system_prompt(model_dir: Path, system_prompt: str) -> bool:
    """Make the project's system prompt the model's default, so `mlx_lm.generate --prompt ...`
    (and any other loader) behaves like the app does. The model was trained with that prompt in
    every example; without it, it answers like the untouched base model. Returns False when the
    template has no default to replace (the README then says to pass --system-prompt)."""
    if not system_prompt.strip():
        return False
    jinja = model_dir / "chat_template.jinja"
    cfg_path = model_dir / "tokenizer_config.json"
    if jinja.exists():
        template, n = _swap_default(jinja.read_text(), system_prompt)
        if n:
            jinja.write_text(template)
        return bool(n)
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if isinstance(cfg.get("chat_template"), str):
            template, n = _swap_default(cfg["chat_template"], system_prompt)
            if n:
                cfg["chat_template"] = template
                cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
            return bool(n)
    return False


def run_command(model_dir: Path | str, system_prompt: str, built_in: bool, prompt: str = "Hello") -> str:
    flag = "" if built_in or not system_prompt else f" --system-prompt {json.dumps(system_prompt)}"
    return f'mlx_lm.generate --model "{model_dir}"{flag} --prompt "{prompt}"'


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


def write_model_card(
    dest: Path, *, name: str, project: dict, lineage: list[dict], sampling: dict, built_in: bool = False
) -> None:
    ram = min_mac_memory_gb(dest)
    system_prompt = project.get("system_prompt") or ""
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
        run_command(dest, system_prompt, built_in),
        "```",
        "",
        (
            "The system prompt below is built into the chat template, so the model uses it whenever no other "
            "system prompt is given."
            if built_in and system_prompt
            else "Pass the system prompt below with --system-prompt (or as the system message): the model was "
            "trained with it in every example and behaves like the base model without it."
            if system_prompt
            else "No system prompt."
        ),
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
            {
                "project": project,
                "lineage": lineage,
                "sampling": sampling,
                "min_ram_gb": ram,
                "system_prompt_built_in": built_in,
            },
            indent=2,
            default=str,
        )
    )


def unique_dest(folder: Path, name: str) -> Path:
    """`folder/name`, or `name-2`, `name-3`… if that's taken, so an export never overwrites another."""
    dest, n = folder / name, 2
    while dest.exists():
        dest, n = folder / f"{name}-{n}", n + 1
    return dest
