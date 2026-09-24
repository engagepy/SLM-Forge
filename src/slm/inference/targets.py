"""Which weights answer a request: current (served) | base | checkpoint:<id> | export:<job id>.

Shared by the Tuner's tools and the playground routes, so both resolve a target the same way and
refuse the same things (a checkpoint or export from another project, a model that isn't there).
"""

from pathlib import Path

from sqlmodel import Session

from slm.db import Checkpoint, Job, Project
from slm.models import manage


class TargetError(ValueError):
    """`status` is the HTTP status the API answers with: 404 (no such thing) or 409 (not ready)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def resolve(s: Session, project: Project, target: str) -> dict:
    """{model_path, adapter_path} for the engine."""
    if target == "base":
        path = manage.local_path_for(project.base_model or "")
        if not path:
            raise TargetError(409, "Base model not downloaded")
        return {"model_path": path, "adapter_path": None}
    if target.startswith("export:"):
        # The finished, exported model exactly as it sits on disk (fused and quantized).
        job = s.get(Job, int(target.split(":", 1)[1]))
        if job is None or job.project_id != project.id or job.kind != "export" or job.status != "succeeded":
            raise TargetError(404, "No such export in this project")
        path = (job.result or {}).get("path")
        if not path or not Path(path).is_dir():
            raise TargetError(409, f"The exported model is no longer at {path}")
        return {"model_path": path, "adapter_path": None}
    if target.startswith("checkpoint:"):
        c = s.get(Checkpoint, int(target.split(":", 1)[1]))
        if c is None or c.project_id != project.id:
            raise TargetError(404, "No such checkpoint in this project")
        if c.fused_path:
            return {"model_path": c.fused_path, "adapter_path": None}
        return {"model_path": c.base_model_path, "adapter_path": c.adapter_path}
    path = manage.serving_path(project)
    if not path:
        raise TargetError(409, "Download the project's base model first")
    return {"model_path": path, "adapter_path": project.current_adapter_path}
