"""Hardware, runtime status and base-model discovery."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from slm import hardware, profile, storage
from slm.config import agent_key_configured, get_settings, tuner_ready
from slm.inference.engine import engine as infer
from slm.models import hub
from slm.train.config import preset
from slm.train.worker import worker

router = APIRouter(prefix="/api", tags=["system"])


def _agent_model(s) -> str:
    if s.agent_provider == "openai":
        return s.openai_model
    return s.claude_model if s.agent_provider == "claude" else s.ollama_model


@router.get("/system")
def system_status() -> dict:
    s = get_settings()
    hw = hardware.detect()
    return {
        "hardware": hw.to_dict(),
        "max_params_4bit": hardware.max_params_for_budget(hw.budget_gb, 4),
        "worker": worker.status(),
        "inference": {"loaded": infer.loaded, "blocked": infer.blocked},
        "agents": {
            "provider": s.agent_provider,
            "model": _agent_model(s),
            "key_configured": agent_key_configured(s),
            "key_env": {"openai": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY", "ollama": None}[s.agent_provider],
            # The Tuner and its specialists are OpenAI-only; the provider switch above covers the
            # judge and the Advanced screens' agents.
            "tuner": {"ready": tuner_ready(s), "key_env": "OPENAI_API_KEY", "model": s.openai_model},
        },
        "workspace": str(s.workspace),
        "disk": storage.footprint(),
    }


@router.get("/huggingface")
def huggingface_account(refresh: bool = False) -> dict:
    """The Hugging Face login this Mac uses (for publishing), and what the GGUF export can make."""
    from slm.models import hub

    return hub.account(refresh)


@router.get("/storage")
def storage_inventory() -> dict:
    """What the app keeps on this Mac, item by item, with sizes."""
    return storage.inventory()


@router.post("/storage/tidy")
def storage_tidy() -> dict:
    """Clear intermediate run files nothing depends on (fused copies, dead runs); adapters, exports
    and every checkpoint's ability to be served again stay."""
    return storage.tidy()


@router.delete("/models/{repo_id:path}")
def remove_model(repo_id: str) -> dict:
    """Delete a downloaded base model from the Hugging Face cache (refused while a project uses it)."""
    try:
        return storage.remove_model(repo_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@router.get("/usage")
def api_usage(refresh: bool = False) -> dict:
    """What the OpenAI account has spent, from OpenAI's Costs API (needs OPENAI_ADMIN_KEY)."""
    from slm import usage

    return usage.spend(refresh=refresh)


@router.get("/models/search")
def search_models(
    q: str = "", mlx_only: bool = True, max_params_b: float | None = None, include_too_big: bool = False
) -> list[dict]:
    try:
        rows = hub.search_models(q, mlx_only=mlx_only, max_params_b=max_params_b, include_too_big=include_too_big)
    except Exception as e:
        raise HTTPException(502, f"Hugging Face search failed: {e}") from e
    return [r.to_dict() for r in rows]


@router.get("/models/inspect")
def inspect_model(repo_id: str) -> dict:
    """Exact shape from config.json plus memory estimates for each training preset."""
    try:
        cfg = hub.fetch_config(repo_id)
    except Exception as e:
        raise HTTPException(404, f"Could not read config.json for {repo_id}: {e}") from e
    shape = hardware.ModelShape.from_config(cfg)
    presets = {}
    for name in ("safe", "balanced", "quality"):
        tc = preset(name, shape)
        presets[name] = {"config": tc.model_dump(), "estimate": tc.memory_estimate(shape).to_dict()}
    return {
        "repo_id": repo_id,
        "params": shape.params,
        "bits": shape.bits,
        "layers": shape.num_layers,
        "hidden_size": shape.hidden_size,
        "vocab_size": shape.vocab_size,
        "inference": hardware.estimate_inference(shape).to_dict(),
        "presets": presets,
        "dpo_safe": preset("safe", shape, "dpo").memory_estimate(shape).to_dict(),
    }


class DownloadReq(BaseModel):
    repo_id: str
    project_id: int | None = None


@router.post("/models/download")
def download_model(req: DownloadReq) -> dict:
    job = worker.submit("download", {"repo_id": req.repo_id}, req.project_id)
    return {"job_id": job.id}


# ── the user profile the Tuner keeps ────────────────────────────────────────


@router.get("/profile")
def get_profile() -> dict:
    return profile.get().model_dump(mode="json")


class LevelIn(BaseModel):
    level: str


@router.post("/profile/level")
def set_profile_level(body: LevelIn) -> dict:
    """The user correcting the Tuner's read of their level."""
    try:
        return profile.set_level(body.level, "set by you").model_dump(mode="json")
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.delete("/profile/notes/{note_id}")
def forget_note(note_id: str) -> dict:
    if not profile.forget(note_id):
        raise HTTPException(404, "no such note")
    return profile.get().model_dump(mode="json")


@router.post("/profile/reset")
def reset_profile() -> dict:
    profile.reset()
    return profile.get().model_dump(mode="json")
