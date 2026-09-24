"""Shared agent plumbing: project context, event logging, proposals."""

from sqlmodel import Session, select

from slm import hardware
from slm.db import DatasetVersion, Project, Proposal, engine, log_event, now
from slm.events import bus


def project_context(project_id: int) -> dict:
    with Session(engine()) as s:
        p = s.get(Project, project_id)
        if p is None:
            raise ValueError(f"project {project_id} not found")
        hw = hardware.detect()
        return {
            "id": p.id,
            "name": p.name,
            "goal": p.goal or "(no goal set)",
            "base_model": p.base_model or "(not chosen yet)",
            "system_prompt": p.system_prompt,
            "hardware": f"{hw.chip}, {hw.total_memory_gb:.0f} GB unified memory (~{hw.budget_gb:.1f} GB usable for ML)",
        }


def event_logger(project_id: int, agent: str):
    def on_event(kind: str, content: dict) -> None:
        log_event(project_id, agent, kind, content)

    return on_event


def create_proposal(project_id: int, agent: str, action: str, title: str, rationale: str, payload: dict) -> Proposal:
    with Session(engine()) as s:
        prop = Proposal(
            project_id=project_id, agent=agent, action=action, title=title, rationale=rationale, payload=payload
        )
        s.add(prop)
        s.commit()
        s.refresh(prop)
    bus.publish(f"project:{project_id}", {"type": "proposal", **prop.model_dump(mode="json")})
    log_event(project_id, agent, "proposal", {"proposal_id": prop.id, "action": action, "title": title})
    return prop


def pending_actions(project_id: int) -> set[str]:
    with Session(engine()) as s:
        rows = s.exec(select(Proposal).where(Proposal.project_id == project_id, Proposal.status == "pending")).all()
        return {r.action for r in rows}


def set_proposal_status(proposal_id: int, status: str, result: dict | None = None) -> Proposal:
    with Session(engine()) as s:
        prop = s.get(Proposal, proposal_id)
        prop.status = status
        if status in ("approved", "rejected"):
            prop.decided_at = now()
        if result is not None:
            prop.result = prop.result | result
        s.add(prop)
        s.commit()
        s.refresh(prop)
    bus.publish(f"project:{prop.project_id}", {"type": "proposal", **prop.model_dump(mode="json")})
    return prop


def eval_prompts(project_id: int) -> set[str]:
    """Normalised prompts from every held-out split and the project's test set: contamination guard."""
    import json
    from pathlib import Path

    from slm.db import test_cases

    prompts: set[str] = set()
    with Session(engine()) as s:
        versions = s.exec(select(DatasetVersion).where(DatasetVersion.project_id == project_id)).all()
        if (p := s.get(Project, project_id)) is not None:
            prompts |= {normalise(c["input"]) for c in test_cases(p)}
    for v in versions:
        for split in ("test", "valid"):
            path = Path(v.path) / f"{split}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                if "messages" in rec:
                    text = next((m["content"] for m in rec["messages"] if m["role"] == "user"), "")
                else:
                    text = rec.get("prompt") or rec.get("text", "")
                prompts.add(normalise(text))
    return prompts


def normalise(text: str) -> str:
    return " ".join(text.lower().split())
