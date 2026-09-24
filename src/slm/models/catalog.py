"""Base models that were designed to be good small models.

A Hub search sorted by downloads surfaces whatever is popular, including cut-down versions of big
models that fine-tune badly at 4-bit. These families were built and trained as small models, run
well on Apple Silicon through mlx-community 4-bit builds (every repo here was checked to exist),
and are what the Tuner shortlists from. Licences matter for what the user can ship, so they're
stated. Task types match the Tuner's plan: persona | qa | extraction | classification | other.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Candidate:
    repo_id: str
    family: str
    params_b: float
    licence: str
    commercial_ok: bool | None  # None: read the licence
    best_for: tuple[str, ...]
    why: str
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


CATALOG: tuple[Candidate, ...] = (
    Candidate(
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "Qwen2.5",
        0.49,
        "Apache-2.0",
        True,
        ("persona", "classification", "extraction", "other"),
        "The strongest tiny instruction follower: keeps a voice and a JSON shape after a few hundred examples.",
    ),  # fmt: skip
    Candidate(
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "Qwen2.5",
        1.54,
        "Apache-2.0",
        True,
        ("qa", "extraction", "classification", "persona"),
        "Explains and extracts noticeably better than 0.5B while still training in minutes.",
    ),  # fmt: skip
    Candidate(
        "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "Qwen2.5",
        3.09,
        "Qwen Research licence",
        False,
        ("extraction", "qa"),
        "Best Qwen2.5 accuracy that fits a 16 GB Mac; multi-field JSON extraction and real explanations.",
        "Research licence: not for commercial use. Prefer Qwen3-4B or SmolLM3 when the user will ship it.",
    ),  # fmt: skip
    Candidate(
        "mlx-community/Qwen3-0.6B-4bit",
        "Qwen3",
        0.60,
        "Apache-2.0",
        True,
        ("persona", "classification", "other"),
        "Newer than Qwen2.5-0.5B, similar size, better at following short formats.",
        "Has a thinking mode: train and run with enable_thinking off for short, fixed outputs.",
    ),  # fmt: skip
    Candidate(
        "mlx-community/Qwen3-1.7B-4bit",
        "Qwen3",
        1.72,
        "Apache-2.0",
        True,
        ("qa", "extraction", "classification"),
        "The sweet spot for Q&A and extraction on a 16 GB Mac: strong, permissive, quick to train.",
        "Thinking mode: keep it off for fixed outputs.",
    ),  # fmt: skip
    Candidate(
        "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "Qwen3",
        4.02,
        "Apache-2.0",
        True,
        ("extraction", "qa"),
        "The most capable permissive model that still trains here (safe preset, short sequences).",
        "Needs the safe preset and batch 1–2 on 16 GB; instruct build, no thinking toggle needed.",
    ),  # fmt: skip
    Candidate(
        "mlx-community/SmolLM2-360M-Instruct",
        "SmolLM2",
        0.36,
        "Apache-2.0",
        True,
        ("persona", "classification"),
        "Built from the ground up as a small model; the lightest option for a voice or a label.",
        "bf16 build (no 4-bit needed at this size); English-centric.",
    ),  # fmt: skip
    Candidate(
        "mlx-community/SmolLM3-3B-4bit",
        "SmolLM3",
        3.08,
        "Apache-2.0",
        True,
        ("qa", "extraction", "persona"),
        "A 3B designed as a small model, permissive, multilingual, long context.",
    ),  # fmt: skip
    Candidate(
        "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "Llama 3.2",
        1.24,
        "Llama 3.2 Community licence",
        True,
        ("qa", "persona"),
        "A well-known general assistant at 1B; good conversational tone.",
        "Community licence: fine for most use, read it for very large products.",
    ),  # fmt: skip
    Candidate(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "Llama 3.2",
        3.21,
        "Llama 3.2 Community licence",
        True,
        ("qa", "extraction"),
        "General knowledge and explanation at 3B.",
    ),  # fmt: skip
    Candidate(
        "mlx-community/gemma-3-1b-it-4bit",
        "Gemma 3",
        1.30,
        "Gemma Terms of Use",
        True,
        ("qa", "persona"),
        "Multilingual, careful tone; a good tutor voice at 1B.",
    ),  # fmt: skip
    Candidate(
        "mlx-community/granite-3.3-2b-instruct-4bit",
        "Granite 3.3",
        2.53,
        "Apache-2.0",
        True,
        ("extraction", "classification", "qa"),
        "Built for enterprise text work: structured output and grounded answers.",
    ),  # fmt: skip
    Candidate(
        "mlx-community/Phi-4-mini-instruct-4bit",
        "Phi-4 mini",
        3.84,
        "MIT",
        True,
        ("qa",),
        "Strong reasoning and maths for its size; heavier to train.",
        "Needs the safe preset on 16 GB.",
    ),  # fmt: skip
)


def recommend(task_type: str | None = None, max_params_b: float | None = None, shipping: bool = False) -> list[dict]:
    """Catalog entries for a task, smallest first within suitability. shipping=True drops
    non-commercial licences."""
    task = (task_type or "other").lower()
    rows = []
    for c in CATALOG:
        if max_params_b and c.params_b > max_params_b:
            continue
        if shipping and c.commercial_ok is False:
            continue
        suited = task in c.best_for
        rows.append((0 if suited else 1, c.params_b, c.to_dict() | {"suited_to_task": suited}))
    return [r[2] for r in sorted(rows, key=lambda r: (r[0], r[1]))]


def lookup(repo_id: str) -> dict | None:
    return next((c.to_dict() for c in CATALOG if c.repo_id == repo_id), None)
