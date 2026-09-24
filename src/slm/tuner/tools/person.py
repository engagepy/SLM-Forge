"""What the Tuner remembers about the person across projects."""

from slm import profile
from slm.tuner.tools._core import Ctx, tool


@tool
def set_user_level(ctx: Ctx, level: str, evidence: str) -> dict:
    """Record how much ML the user knows: beginner | intermediate | expert. Infer it from how they
    write: vocabulary, what they ask about, what they want to control. For example, asking about
    LoRA rank, learning-rate schedules or DPO β suggests expert; "what's a model?" suggests
    beginner. evidence: one short line saying why. It's remembered for all their future projects,
    and you'll adapt your explanations to it. Update it if new evidence contradicts it."""
    p = profile.set_level(level, evidence)
    return {"level": p.level, "evidence": p.level_evidence}


@tool
def remember_about_user(ctx: Ctx, text: str, kind: str = "preference") -> dict:
    """Remember something durable about the user for all their future projects. kind: preference
    (e.g. "prefers the smallest model that works", "wants 4-bit exports", "likes to judge answers
    themselves"), fact (e.g. "teaches high-school physics"), or goal (e.g. "wants to ship models
    to an iPhone app"). Only things that will still be true next time; not project details."""
    return profile.remember(text, kind, ctx.context.project_id)
