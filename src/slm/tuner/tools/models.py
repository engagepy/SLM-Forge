"""Choosing a base model."""

from sqlmodel import Session

from slm import hardware
from slm.db import (
    Project,
    engine,
)
from slm.models import catalog, hub, manage
from slm.tuner import confirm
from slm.tuner.tools._core import Ctx, tool


@tool
def find_base_models(ctx: Ctx, query: str = "", max_params_billion: float = 3.0, task_type: str | None = None) -> dict:
    """Base models for this project. "recommended": a curated catalog of families built as small
    models (with licence, what each is best for, why, and whether it's already on this Mac),
    ordered by suitability to task_type (persona | qa | extraction | classification | other) then
    size; every entry exists as an MLX 4-bit build. "more_from_hub": a Hub search for `query`, for
    when the user names a model. Shortlist 2–3 from recommended, then choose_base_model."""
    local = {m["repo_id"]: m for m in manage.local_models()}
    budget = hardware.detect().budget_gb
    recommended = []
    for c in catalog.recommend(task_type, max_params_b=max_params_billion):
        train_gb = hub.rough_training_gb(int(c["params_b"] * 1e9), 4)
        recommended.append(
            c
            | {
                "train_memory_gb": round(train_gb, 2),
                "fit": hub.fit_verdict(train_gb, budget),
                "already_on_this_mac": c["repo_id"] in local,
            }
        )
    recommended.sort(key=lambda m: (not m["suited_to_task"], not m["already_on_this_mac"], m["params_b"]))
    rows = hub.search_models(query, max_params_b=max_params_billion, limit=12) if query else []
    found = {
        m.id: {
            "repo_id": m.id,
            "params_b": round(m.params / 1e9, 2),
            "bits": m.bits,
            "train_memory_gb": m.train_estimate_gb,
            "fit": m.fit,
            "downloads": m.downloads,
            "already_on_this_mac": m.id in local,
        }
        for m in rows
    }
    words = [w for w in query.lower().split() if len(w) > 2]
    for rid, m in local.items():  # local models matching the query, even if the Hub search missed them
        if (
            rid not in found
            and m["params_b"] <= max_params_billion
            and (not words or any(w in rid.lower() for w in words))
        ):
            found[rid] = {k: m[k] for k in ("repo_id", "params_b", "bits", "train_memory_gb", "fit")} | {
                "already_on_this_mac": True
            }
    more = sorted(found.values(), key=lambda m: (not m["already_on_this_mac"], m["params_b"]))
    return {
        "recommended": recommended,
        "more_from_hub": [m for m in more if m["repo_id"] not in {r["repo_id"] for r in recommended}],
    }


@tool
def choose_base_model(ctx: Ctx, repo_id: str, reason: str) -> dict:
    """Propose a base model. The user confirms it on a card; only then is it set (and downloaded
    if it isn't on this Mac yet). reason: one or two plain sentences on why this model, shown on
    the card. Only possible before any training has happened."""
    pid = ctx.context.project_id
    with Session(engine()) as s:
        if confirm.base_model_locked(s, s.get(Project, pid), repo_id):
            raise ValueError(confirm.SWITCH_BASE)
    on_mac = manage.local_path_for(repo_id) is not None or manage.register_local(repo_id) is not None
    info = next((m for m in manage.local_models() if m["repo_id"] == repo_id), None) if on_mac else None
    details = {"repo_id": repo_id, "on_this_mac": on_mac}
    if info:
        details |= {"params_b": info["params_b"], "bits": info["bits"]}
    # What the user may do with a model built on this base: on the card, before they say go.
    lic = catalog.licence_for(repo_id, manage.local_path_for(repo_id))
    details |= {"licence": lic["licence"], "licence_url": lic["url"], "licence_conditions": lic["conditions"],
                "commercial_ok": lic["commercial_ok"]}  # fmt: skip
    title = f"Use {repo_id.split('/')[-1]}" + ("" if on_mac else " (download it)")
    return confirm.propose(pid, "model", title, reason, details, {"repo_id": repo_id})
