"""Build specs: a YAML recipe (`kind: slm_build`) turned into an SLM Forge project.

A spec describes a whole build (goal, output contract, data, training, eval gates, publishing),
often written for a cloud GPU. SLM Forge keeps what it can do on this Mac: the name, goal, the
output contract as the system prompt, and a plan (data sources, eval, stop rule) for the Tuner,
which then runs the build the usual way, every run waiting for the user's go-ahead. Everything
else becomes a warning, so the user sees what was dropped instead of it being silently ignored.
"""

import json

import yaml

from slm.tuner.tools._core import MAX_SYNTHETIC_TOTAL

TASK_TYPES = {
    "structured_extraction": "extraction",
    "extraction": "extraction",
    "classification": "classification",
    "qa": "qa",
    "persona": "persona",
}
SAFE_LR = 1e-4  # AGENTS.md lesson 6: 2e-4 diverged


def _get(d: dict, path: str, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _system_prompt(spec: dict) -> str:
    c = spec["task_contract"]
    out = c.get("output") or {}
    parts = [str(_get(spec, "meta.one_liner", "")).strip()]
    if desc := _get(c, "input.description"):
        parts.append(f"Input: {desc}")
    if out.get("format") == "json" and out.get("json_schema"):
        schema = json.dumps(out["json_schema"], separators=(",", ":"))
        parts.append(f"Reply with JSON only, matching this JSON Schema: {schema}")
    elif out.get("format"):
        parts.append(f"Reply in {out['format']} only.")
    if rules := out.get("invariants"):
        parts.append("Rules: " + "; ".join(map(str, rules)) + ".")
    return "\n".join(p for p in parts if p)


def _warnings(spec: dict) -> list[str]:
    w = []
    acc = _get(spec, "compute.accelerator") or {}
    if _get(spec, "compute.provider") or acc:
        w.append(
            f"Compute: the spec asks for {acc.get('count', 1)}× {acc.get('type', 'GPU')} on "
            f"{_get(spec, 'compute.provider', 'a cloud provider')}; SLM Forge trains on this Mac with MLX."
        )
    if bm := _get(spec, "base_model.id"):
        w.append(f"Base model: the spec asks for {bm}; the Tuner proposes the largest model this Mac trains well.")
    phases = _get(spec, "training.phases") or []
    if any(p.get("method") == "full" for p in phases):
        w.append("Training: full fine-tuning isn't supported; SLM Forge trains LoRA adapters.")
    if any(p.get("sweep") for p in phases):
        w.append("Training: hyper-parameter sweeps aren't run; the Tuner trains one round at a time.")
    lrs = [p["lr"] for p in phases if isinstance(p.get("lr"), (int, float))]
    if any(lr > SAFE_LR for lr in lrs):
        w.append(f"Training: learning rates above {SAFE_LR:g} diverge on small models; the Tuner uses {SAFE_LR:g}.")
    if fw := _get(spec, "training.framework"):
        w.append(f"Training: {fw} isn't used; SLM Forge trains with mlx-lm.")
    if _get(spec, "training.logging.backend"):
        w.append(f"Logging: {_get(spec, 'training.logging.backend')} isn't used; metrics show in the Studio.")
    n = _get(spec, "data.synthetic.target_rows", 0)
    if _get(spec, "data.synthetic.enabled") and n > MAX_SYNTHETIC_TOTAL:
        w.append(
            f"Synthetic data: the spec asks for {n} rows; SLM Forge writes at most {MAX_SYNTHETIC_TOTAL} "
            "(seeds and top-ups) and builds the dataset from public data."
        )
    if _get(spec, "data.synthetic.teacher.id") or _get(spec, "data.synthetic.teacher.endpoint"):
        w.append("Teacher model: the spec's teacher isn't used; the agent provider writes examples.")
    if (eng := _get(spec, "inference.engine")) and eng != "mlx":
        w.append(f"Inference: {eng} isn't used; the model runs with MLX.")
    if _get(spec, "inference.structured_decoding.enabled"):
        w.append("Inference: constrained (structured) decoding isn't available; the output format is learned.")
    for b in _get(spec, "eval.baselines") or []:
        if b.get("mode") == "zero_shot" and str(b.get("id", "")).startswith("${"):
            w.append(f"Evaluation: baseline {b.get('label', b['id'])} needs an external model and isn't run.")
    if _get(spec, "eval.harness"):
        w.append(f"Evaluation: {_get(spec, 'eval.harness')} isn't run; evaluate_model scores a fixed test set.")
    for v in _get(spec, "quantization.variants") or []:
        if v.get("format") != "gguf":
            w.append(f"Quantization: {v.get('format')} isn't supported; exports are MLX, with optional GGUF.")
    if _get(spec, "publish.enabled"):
        w.append("Publishing: never automatic; publish an export to Hugging Face yourself when you're happy with it.")
    if _get(spec, "pipeline.stages"):
        w.append("Pipeline: stages aren't run unattended; the Tuner proposes each run for you to confirm.")
    if spec.get("next_builds"):
        w.append(f"next_builds: {len(spec['next_builds'])} queued builds are ignored; import each spec on its own.")
    return w


def load(text: str) -> dict:
    """Parse a spec. Returns {project: {name, goal, system_prompt, plan}, warnings}. Raises ValueError."""
    try:
        spec = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"Not valid YAML: {e}") from e
    if not isinstance(spec, dict) or spec.get("kind") != "slm_build":
        raise ValueError("Not a build spec: expected `kind: slm_build` at the top level.")
    if not isinstance(spec.get("task_contract"), dict) or not _get(spec, "meta.one_liner"):
        raise ValueError("A build spec needs `meta.one_liner` and a `task_contract`.")

    gold = _get(spec, "data.gold") or []
    sources = [
        f"{g['source']['id']} ({g.get('license', 'licence unknown')}; {'/'.join(g.get('role', []))})"
        for g in gold
        if _get(g, "source.type") == "hf_dataset" and _get(g, "source.id")
    ]
    test = _get(spec, "data.splits.test") or {}
    metrics = [m["name"] for m in _get(spec, "eval.metrics") or [] if isinstance(m, dict) and "name" in m]
    gates = [str(g) for g in _get(spec, "eval.gates") or [] if "baselines." not in str(g)]
    plan = {
        "source": f"spec:{spec.get('id', 'unnamed')}",
        "task_type": TASK_TYPES.get(_get(spec, "meta.task_family", ""), "other"),
        "output_format": (
            "JSON matching the schema in the system prompt"
            if _get(spec, "task_contract.output.format") == "json"
            else str(_get(spec, "task_contract.output.format", "as the system prompt says"))
        ),
        "model_tier": f"largest this Mac trains comfortably (spec asked for {_get(spec, 'base_model.id', 'any')})",
        "data_sources": "; ".join(sources) or "find public data on Hugging Face",
        "data_target": "public data mapped into the output format; synthetic examples only for gaps",
        "eval_design": (
            f"30–60 exact-match cases held out from {test.get('from', 'the gold data')} before any training data"
            + (f"; spec metrics: {', '.join(metrics)}" if metrics else "")
        ),
        "stop_rule": "; ".join(gates) or "test score stops improving",
    }
    plan = {k: v[:400] for k, v in plan.items()}
    project = {
        "name": str(_get(spec, "meta.display_name") or spec.get("id") or "Imported spec")[:60],
        "goal": " ".join(str(spec["meta"]["one_liner"]).split()),
        "system_prompt": _system_prompt(spec),
        "plan": plan,
    }
    return {"project": project, "warnings": _warnings(spec)}
