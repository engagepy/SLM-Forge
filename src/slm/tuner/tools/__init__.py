"""The Tuner's hands: every action it can take on the platform, one module per area.

Each tool wraps existing, tested backend code and returns a compact dict the agent can reason
about. Quick jobs (dataset import, preparation, synthesis) are awaited inside the tool. Runs
(downloads, training, export) are only proposed: the user confirms them (tuner/confirm.py), and the
Tuner is woken when they're confirmed and again when they finish.
"""

from slm.tuner.tools._core import MAX_UNREVIEWED, REGISTRY, Ctx, TunerContext, _spend, _submit, _wait, tool
from slm.tuner.tools.data import (
    build_dataset_from_examples,
    generate_synthetic_examples,
    import_dataset,
    inspect_dataset,
    plan_preparation,
    prepare_dataset,
    preview_dataset,
    review_synthetic_examples,
    scout_datasets,
    search_datasets,
)
from slm.tuner.tools.evaluation import (
    ai_review_answers,
    ask_user_to_compare,
    evaluate_model,
    feedback_summary,
    finish_project,
    try_model,
)
from slm.tuner.tools.models import choose_base_model, find_base_models
from slm.tuner.tools.person import remember_about_user, set_user_level
from slm.tuner.tools.status import get_status, machine_overview, manage_project, set_stage, update_project
from slm.tuner.tools.training import (
    cancel_job,
    export_model,
    plan_training,
    serve_checkpoint,
    start_training,
    training_progress,
)

# In the order the agent sees them. Every @tool (REGISTRY) must be here: a test guards it.
ALL_TOOLS = [
    get_status,
    set_stage,
    update_project,
    find_base_models,
    choose_base_model,
    search_datasets,
    preview_dataset,
    scout_datasets,
    plan_preparation,
    import_dataset,
    inspect_dataset,
    prepare_dataset,
    plan_training,
    start_training,
    training_progress,
    cancel_job,
    try_model,
    evaluate_model,
    serve_checkpoint,
    ask_user_to_compare,
    feedback_summary,
    generate_synthetic_examples,
    review_synthetic_examples,
    build_dataset_from_examples,
    export_model,
    ai_review_answers,
    finish_project,
    machine_overview,
    manage_project,
    set_user_level,
    remember_about_user,
]

__all__ = ["ALL_TOOLS", "REGISTRY", "MAX_UNREVIEWED", "Ctx", "TunerContext", "tool", "_spend", "_submit", "_wait"]
