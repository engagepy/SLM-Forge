"""Turn one human judgement into training signal. Shared by the API and the Studio canvas."""

from sqlmodel import Session

from slm.db import Feedback, PreferencePair, SftExample, engine

CHOICES = ("a", "b", "tie", "both_bad")


def record_feedback(
    project_id: int,
    *,
    prompt: str,
    candidate_a: str,
    candidate_b: str,
    choice: str,
    system: str = "",
    edited_answer: str = "",
    critique: str = "",
    params_a: dict | None = None,
    params_b: dict | None = None,
    model_ref: str = "",
) -> dict:
    """Store the judgement and derive training signal from it.

    - a/b picked → preference pair (picked vs other)
    - edited answer → SFT example, plus pairs of (edit vs each candidate the human didn't pick)
    - tie with no edit → no training signal, just context for the Observer
    """
    if choice not in CHOICES:
        raise ValueError("choice must be a, b, tie or both_bad")
    if choice == "both_bad" and not edited_answer.strip() and not critique.strip():
        raise ValueError("When both are bad, write a better answer or a critique")
    with Session(engine()) as s:
        fb = Feedback(
            project_id=project_id,
            prompt=prompt,
            system=system,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            params_a=params_a or {},
            params_b=params_b or {},
            choice=choice,
            edited_answer=edited_answer,
            critique=critique,
            model_ref=model_ref,
        )
        s.add(fb)
        s.commit()
        s.refresh(fb)

        pairs, sft = 0, 0
        cands = {"a": candidate_a, "b": candidate_b}
        edited = edited_answer.strip()

        def pair(chosen: str, rejected: str) -> None:
            nonlocal pairs
            if chosen.strip() and rejected.strip() and chosen.strip() != rejected.strip():
                s.add(
                    PreferencePair(
                        project_id=project_id,
                        prompt=prompt,
                        system=system,
                        chosen=chosen,
                        rejected=rejected,
                        source="human",
                        feedback_id=fb.id,
                    )
                )
                pairs += 1

        if edited:
            rejected_side = [k for k in cands if k != choice] if choice in ("a", "b") else list(cands)
            for k in rejected_side:
                pair(edited, cands[k])
            msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": edited}]
            if system:
                msgs.insert(0, {"role": "system", "content": system})
            s.add(SftExample(project_id=project_id, messages=msgs, source="feedback", feedback_id=fb.id))
            sft += 1
        elif choice in ("a", "b"):
            other = "b" if choice == "a" else "a"
            pair(cands[choice], cands[other])
        s.commit()
        return {"feedback_id": fb.id, "preference_pairs": pairs, "sft_examples": sft}
