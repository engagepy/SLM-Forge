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


QUANTIZE_BITS = (3, 4, 6, 8)


def check_bits(bits: int | None) -> int | None:
    if bits is not None and bits not in QUANTIZE_BITS:
        raise ValueError(f"quantize_bits must be one of {', '.join(map(str, QUANTIZE_BITS))}")
    return bits


def quantize_command(src: Path, dest: Path, bits: int, group_size: int = 64) -> list[str]:
    return [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", str(src),
        "--mlx-path", str(dest),
        "-q", "--q-bits", str(bits), "--q-group-size", str(group_size),
    ]  # fmt: skip


# Chat templates disagree on what happens without a system message: Qwen2.5 prints a fixed
# sentence, SmolLM and Granite build one in code, Qwen3, Llama, Gemma and Phi print nothing. Rather
# than rewriting each family's default, the bake prepends one block that makes the project's prompt
# messages[0] whenever the caller gave no system message. Every template then follows its own
# "system message given" path, and a caller's explicit system prompt still wins.
_BAKE_OPEN = "{#- slm-forge: the model's system prompt, used when the caller gives none -#}"
_BAKE_CLOSE = "{#- /slm-forge -#}"
_BAKED = re.compile(re.escape(_BAKE_OPEN) + ".*?" + re.escape(_BAKE_CLOSE), re.S)


def _jinja_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'").replace("\r", "\\r").replace("\n", "\\n")


def _inject_default(template: str, system_prompt: str) -> str:
    block = (
        f"{_BAKE_OPEN}{{%- if messages and messages[0]['role'] != 'system' %}}"
        f"{{%- set messages = [{{'role': 'system', 'content': '{_jinja_literal(system_prompt)}'}}] + messages %}}"
        f"{{%- endif %}}{_BAKE_CLOSE}"
    )
    return block + _BAKED.sub("", template, count=1)


def renders_with_default(template: str, system_prompt: str) -> bool:
    """Render the template the way tokenizers do and check the prompt reaches the model text."""
    from transformers.utils.chat_template_utils import render_jinja_template

    try:
        rendered, _ = render_jinja_template(
            conversations=[[{"role": "user", "content": "Hello"}]],
            chat_template=template,
            add_generation_prompt=True,
            bos_token="<s>",
            eos_token="</s>",
        )
    except Exception:
        return False
    return system_prompt.strip() in rendered[0]


def bake_system_prompt(model_dir: Path, system_prompt: str) -> bool:
    """Make the project's system prompt the model's default, so `mlx_lm.generate --prompt ...`
    (and any other loader) behaves like the app does. The model was trained with that prompt in
    every example; without it, it answers like the untouched base model. The change is verified
    by rendering; returns False (template untouched) when it can't be, and the README then says to
    pass --system-prompt. Baking again replaces the earlier prompt."""
    if not system_prompt.strip():
        return False
    jinja = model_dir / "chat_template.jinja"
    cfg_path = model_dir / "tokenizer_config.json"
    if jinja.exists():
        template = _inject_default(jinja.read_text(), system_prompt)
        if not renders_with_default(template, system_prompt):
            return False
        jinja.write_text(template)
        return True
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if isinstance(cfg.get("chat_template"), str):
            template = _inject_default(cfg["chat_template"], system_prompt)
            if not renders_with_default(template, system_prompt):
                return False
            cfg["chat_template"] = template
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
            return True
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
