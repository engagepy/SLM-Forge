"""Run mlx_lm / mlx_lm_lora trainers as subprocesses and parse their progress output.

Running out-of-process means a crashed or OOM-killed run can't take the server down, and
all GPU memory is returned when the job exits.
"""

import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)"

# mlx_lm.lora
_SFT_TRAIN = re.compile(
    rf"Iter (\d+): Train loss {_NUM}, Learning Rate {_NUM}, It/sec {_NUM}, "
    rf"Tokens/sec {_NUM}, Trained Tokens (\d+), Peak mem {_NUM} GB"
)
_SFT_VAL = re.compile(rf"Iter (\d+): Val loss {_NUM}, Val took {_NUM}s")

# mlx_lm_lora (DPO)
_DPO_TRAIN = re.compile(
    rf"Iter (\d+): loss {_NUM}, chosen_r {_NUM}, rejected_r {_NUM}, acc {_NUM}, "
    rf"margin {_NUM}, lr {_NUM}, it/s {_NUM}, tok/s {_NUM}, peak_mem {_NUM}GB"
)
_DPO_VAL = re.compile(
    rf"Iter (\d+): Val loss {_NUM}, Val chosen reward {_NUM}, Val rejected reward {_NUM}, "
    rf"Val accuracy {_NUM}, Val margin {_NUM}, Val took {_NUM}s"
)
_TOTAL_ITERS = re.compile(r"Starting training\.*,? iters: (\d+)")


@dataclass
class ParsedMetric:
    iteration: int
    split: str  # train | val
    values: dict


def _f(s: str) -> float:
    return float(s)


def parse_line(line: str) -> ParsedMetric | None:
    line = _ANSI.sub("", line).strip()
    if m := _SFT_TRAIN.search(line):
        it, loss, lr, its, tps, toks, mem = m.groups()
        return ParsedMetric(
            int(it),
            "train",
            {
                "loss": _f(loss),
                "learning_rate": _f(lr),
                "it_per_sec": _f(its),
                "tokens_per_sec": _f(tps),
                "trained_tokens": int(toks),
                "peak_mem_gb": _f(mem),
            },
        )
    if m := _SFT_VAL.search(line):
        it, loss, took = m.groups()
        return ParsedMetric(int(it), "val", {"loss": _f(loss), "val_time_s": _f(took)})
    if m := _DPO_TRAIN.search(line):
        it, loss, cr, rr, acc, margin, lr, its, tps, mem = m.groups()
        return ParsedMetric(
            int(it),
            "train",
            {
                "loss": _f(loss),
                "chosen_reward": _f(cr),
                "rejected_reward": _f(rr),
                "accuracy": _f(acc),
                "margin": _f(margin),
                "learning_rate": _f(lr),
                "it_per_sec": _f(its),
                "tokens_per_sec": _f(tps),
                "peak_mem_gb": _f(mem),
            },
        )
    if m := _DPO_VAL.search(line):
        it, loss, cr, rr, acc, margin, took = m.groups()
        return ParsedMetric(
            int(it),
            "val",
            {
                "loss": _f(loss),
                "chosen_reward": _f(cr),
                "rejected_reward": _f(rr),
                "accuracy": _f(acc),
                "margin": _f(margin),
                "val_time_s": _f(took),
            },
        )
    return None


def parse_total_iters(line: str) -> int | None:
    m = _TOTAL_ITERS.search(_ANSI.sub("", line))
    return int(m.group(1)) if m else None


def trainer_command(mode: str, config_path: Path) -> list[str]:
    if mode == "sft":
        return [sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_path)]
    return [sys.executable, "-m", "mlx_lm_lora.train", "-c", str(config_path), "--train"]


class Cancelled(Exception):
    pass


# Warnings the trainer repeats once per batch; logged once with a count instead.
_NOISY = [(re.compile(r"^\[WARNING\] Some sequences are longer than \d+ tokens"), "sequence truncated")]


def _noise_key(line: str) -> str | None:
    return next((key for pattern, key in _NOISY if pattern.search(line)), None)


_SUPPRESSED = re.compile(r"^\[(\d+) more 'sequence truncated' warnings suppressed\]")


def truncation_count(line: str) -> int:
    """How many truncation warnings this output line stands for: the first one, or the summary."""
    if _noise_key(line) == "sequence truncated":
        return 1
    if m := _SUPPRESSED.match(line):
        return int(m.group(1))
    return 0


def run_process(
    cmd: list[str],
    *,
    log_path: Path,
    on_line: Callable[[str], None],
    should_cancel: Callable[[], bool],
    cwd: Path | None = None,
) -> int:
    """Stream a subprocess's merged output line by line, honouring cancellation."""
    env = os.environ | {
        "PYTHONUNBUFFERED": "1",
        "NO_COLOR": "1",
        "TQDM_DISABLE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=cwd,
        )
        assert proc.stdout is not None
        cancelled = threading.Event()

        def watch() -> None:
            # Poll independently of output, so a silent phase (model load) is still cancellable.
            while proc.poll() is None:
                if should_cancel():
                    cancelled.set()
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return
                time.sleep(0.5)

        threading.Thread(target=watch, daemon=True).start()
        repeats: dict[str, int] = {}
        for raw in proc.stdout:
            # Progress bars redraw with \r; keep only the final state of each line.
            line = raw.rstrip("\n").split("\r")[-1]
            if (key := _noise_key(line)) is not None:
                repeats[key] = repeats.get(key, 0) + 1
                if repeats[key] > 1:
                    continue  # show the first occurrence only; summarised at the end
            log.write(line + "\n")
            log.flush()
            on_line(line)
        for key, n in repeats.items():
            if n > 1:
                summary = f"[{n - 1} more '{key}' warnings suppressed]"
                log.write(summary + "\n")
                on_line(summary)
        code = proc.wait()
        if cancelled.is_set():
            raise Cancelled()
        return code


def write_config(cfg: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
