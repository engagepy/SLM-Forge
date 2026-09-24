"""The local user profile: expertise level and remembered preferences, shared across projects.

The Tuner updates it as it learns about the user and reads it at the start of every turn, so a
new project starts already knowing them. Stored locally in SQLite; note that it's included in
the Tuner's prompts, so it is sent to the agent's LLM provider along with the conversation.
"""

import uuid

from sqlmodel import Session

from slm.db import UserProfile, engine, now

LEVELS = ("unknown", "beginner", "intermediate", "expert")
NOTE_KINDS = ("preference", "fact", "goal")
MAX_NOTES = 40

STYLE = {
    "unknown": (
        "You don't know their level yet. Start plain and friendly, without being condescending; "
        "watch how they write (vocabulary, questions, what they ask to control) and record their "
        "level with set_user_level as soon as the evidence is clear."
    ),
    "beginner": (
        "A beginner. Use plain language and everyday analogies, no unexplained jargon, one idea at "
        "a time. Keep full autopilot: decide everything and summarise the outcome, not the mechanics. "
        "Reassure them it's working and what they'll end up with."
    ),
    "intermediate": (
        "Knows the basics. Name the techniques (LoRA fine-tuning, DPO, validation loss) with a "
        "one-line why, and show the key numbers. Stay on autopilot, but mention the one or two "
        "choices they might want to change."
    ),
    "expert": (
        "An ML expert (research or engineering level). Be technical and dense: LoRA rank/scale and "
        "layers tuned, learning-rate schedule and warmup, effective batch size, token-length "
        "distribution and truncation, eval methodology and contamination, DPO β and loss variant, "
        "quantisation trade-offs. Show your reasoning briefly and cite the numbers from tools. "
        "Accept their overrides and discuss trade-offs as a peer; don't explain the basics. Keep "
        "autopilot moving unless they want to drive a step themselves."
    ),
}


def get() -> UserProfile:
    with Session(engine()) as s:
        p = s.get(UserProfile, 1)
        if p is None:
            p = UserProfile(id=1)
            s.add(p)
            s.commit()
            s.refresh(p)
        return p


def set_level(level: str, evidence: str = "") -> UserProfile:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    with Session(engine()) as s:
        p = s.get(UserProfile, 1) or UserProfile(id=1)
        p.level, p.level_evidence, p.updated_at = level, evidence[:300], now()
        s.add(p)
        s.commit()
        s.refresh(p)
        return p


def remember(text: str, kind: str = "preference", project_id: int | None = None) -> dict:
    """Add a note, replacing an existing one with the same text (case-insensitive)."""
    if kind not in NOTE_KINDS:
        raise ValueError(f"kind must be one of {NOTE_KINDS}")
    text = " ".join(text.split())[:240]
    if not text:
        raise ValueError("empty note")
    with Session(engine()) as s:
        p = s.get(UserProfile, 1) or UserProfile(id=1)
        notes = [n for n in p.notes if n["text"].lower() != text.lower()]
        note = {
            "id": uuid.uuid4().hex[:8],
            "kind": kind,
            "text": text,
            "project_id": project_id,
            "created_at": now().isoformat(),
        }
        p.notes = (notes + [note])[-MAX_NOTES:]  # reassigned, so the JSON change persists
        p.updated_at = now()
        s.add(p)
        s.commit()
        return note


def forget(note_id: str) -> bool:
    with Session(engine()) as s:
        p = s.get(UserProfile, 1)
        if p is None:
            return False
        kept = [n for n in p.notes if n["id"] != note_id]
        changed = len(kept) != len(p.notes)
        p.notes, p.updated_at = kept, now()
        s.add(p)
        s.commit()
        return changed


def reset() -> None:
    with Session(engine()) as s:
        p = s.get(UserProfile, 1) or UserProfile(id=1)
        p.level, p.level_evidence, p.notes, p.updated_at = "unknown", "", [], now()
        s.add(p)
        s.commit()


def prompt_section(p: UserProfile | None = None) -> str:
    """The part of the Tuner's instructions that describes who it's working with."""
    p = p or get()
    lines = [
        "## Who you're working with",
        f"Level: **{p.level}**" + (f" ({p.level_evidence})" if p.level_evidence else ""),
    ]
    lines.append(STYLE.get(p.level, STYLE["unknown"]))
    if p.notes:
        lines.append("What you've learned about them in earlier sessions (apply it; don't re-ask):")
        lines += [f"- [{n['kind']}] {n['text']}" for n in p.notes[-25:]]
    lines.append(
        "Whatever their level, the objective is the same: get them to their own working, exported SLM. "
        "Adapt how you explain and how much you involve them, never whether you finish."
    )
    return "\n".join(lines)
