"""Asking the local model to answer: which weights, one answer, two answers to compare."""

import random

from sqlmodel import Session

from slm.db import (
    Project,
    engine,
)
from slm.inference import targets
from slm.inference.engine import EngineBusy, SamplingParams
from slm.inference.engine import engine as infer


def _where(project: Project, target: str) -> dict:
    """Which weights answer: current (served) | base (untrained) | checkpoint:<id>."""
    with Session(engine()) as s:
        return targets.resolve(s, project, target)  # a TargetError is a ValueError the model reads


def _generate(
    project: Project, prompt: str, temperature: float, max_tokens: int, target: str = "current", seed: int | None = None
) -> dict:
    where = _where(project, target)
    messages = [{"role": "user", "content": prompt}]
    if project.system_prompt:
        messages.insert(0, {"role": "system", "content": project.system_prompt})
    try:
        text, stats = infer.generate(
            messages, SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed), **where
        )
    except EngineBusy as e:
        raise ValueError(f"The GPU is busy ({e}); try again when the job finishes.") from e
    return {"text": text.strip(), "tokens_per_sec": stats.get("tokens_per_sec"), "finish": stats.get("finish_reason")}


# Two samplings of the same prompt: a steady one and an adventurous one, so they differ enough to compare.
PARAMS_A, PARAMS_B = {"temperature": 0.7}, {"temperature": 1.05}


def _two_answers(project: Project, prompt: str, max_tokens: int) -> tuple[str, str]:
    return tuple(
        _generate(project, prompt, p["temperature"], max_tokens, seed=random.randint(0, 10**9))["text"]
        for p in (PARAMS_A, PARAMS_B)
    )
