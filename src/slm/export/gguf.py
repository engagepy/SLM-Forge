"""Convert an export to GGUF, the format llama.cpp, Ollama and LM Studio run.

mlx-lm's own GGUF writer only handles Llama-family models, so this uses llama.cpp's converter. It
needs torch and an older transformers than SLM Forge does, so it runs in its own virtualenv,
downloaded and installed once into the workspace (`tools/`, counted by the disk meter). The
converter writes F16 and Q8_0 itself; Q4_K_M needs llama.cpp's `llama-quantize` binary
(`brew install llama.cpp`), used when it is installed.
"""

import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from slm.config import get_settings

LLAMA_CPP_TAG = "v0.5.0"
SOURCE_URL = f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{LLAMA_CPP_TAG}.tar.gz"
DIRECT = {"Q8_0": "q8_0", "F16": "f16", "BF16": "bf16"}  # what the converter writes itself
QUANTIZER = {"Q4_K_M", "Q5_K_M", "Q6_K", "Q4_0"}  # what needs llama-quantize
QUANTS = tuple(DIRECT) + tuple(sorted(QUANTIZER))
BREW_HINT = "Q4_K_M needs llama.cpp's quantizer: install it with `brew install llama.cpp`, then make the GGUF again."


def toolchain_dir() -> Path:
    return get_settings().workspace / "tools" / f"llama.cpp-{LLAMA_CPP_TAG}"


def _src() -> Path:
    return toolchain_dir() / "src"


def _python() -> Path:
    return toolchain_dir() / "venv" / "bin" / "python"


def toolchain_ready() -> bool:
    return (_src() / "convert_hf_to_gguf.py").is_file() and (toolchain_dir() / "venv" / ".installed").is_file()


def ensure_toolchain(note=print, run=subprocess.run) -> None:
    """Download llama.cpp's converter and install its dependencies, once. `run` is swappable for
    tests. Raises with a readable message on failure; a half-finished install is redone next time."""
    if toolchain_ready():
        return
    root = toolchain_dir()
    if not (_src() / "convert_hf_to_gguf.py").is_file():
        note(f"First GGUF export: downloading llama.cpp {LLAMA_CPP_TAG}'s converter (one time) ...")
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "src.tar.gz"
        try:
            urllib.request.urlretrieve(SOURCE_URL, archive)
        except OSError as e:
            raise RuntimeError(f"Couldn't download llama.cpp's converter ({e}). Check the connection and retry.") from e
        with tarfile.open(archive) as tar:
            top = tar.getnames()[0].split("/")[0]
            tar.extractall(root, filter="data")
        shutil.rmtree(_src(), ignore_errors=True)
        (root / top).rename(_src())
        archive.unlink()
    note("Installing the converter's dependencies (torch and transformers, about 300 MB, one time) ...")
    venv = root / "venv"
    shutil.rmtree(venv, ignore_errors=True)
    steps = [
        [sys.executable, "-m", "venv", str(venv)],
        [str(_python()), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        [str(_python()), "-m", "pip", "install", "--quiet", "-r",
         str(_src() / "requirements" / "requirements-convert_hf_to_gguf.txt")],
    ]  # fmt: skip
    for cmd in steps:
        r = run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Installing the GGUF converter failed: {(r.stderr or r.stdout)[-800:]}")
    (venv / ".installed").write_text(LLAMA_CPP_TAG)
    note("GGUF converter ready.")


def quantizer() -> str | None:
    return shutil.which("llama-quantize")


def convert_command(model_dir: Path, out: Path, outtype: str) -> list[str]:
    return [str(_python()), str(_src() / "convert_hf_to_gguf.py"), str(model_dir),
            "--outfile", str(out), "--outtype", outtype]  # fmt: skip


def quantize_command(source: Path, out: Path, quant: str) -> list[str]:
    return [quantizer() or "llama-quantize", str(source), str(out), quant]


def gguf_name(export_name: str, quant: str) -> str:
    return f"{export_name}-{quant}.gguf"
