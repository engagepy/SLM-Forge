"""The job handlers (sft, dpo, export) against a fake trainer: what they write, record and chain.

The fake stands in for the mlx_lm subprocesses: it reads the YAML the handler wrote, creates the
adapter files the trainer would, and prints progress lines in the trainers' real formats so the
metrics parser and the diagnostics run for real.
"""

import json
import shutil
from pathlib import Path

import pytest
import yaml
from sqlmodel import select

from slm.db import Checkpoint, DatasetVersion, Job, Metric, ModelRecord, Project
from slm.train import jobs, runner
from slm.train.worker import JobContext, worker

FIXTURES = Path(__file__).parent / "fixtures" / "templates"
CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "vocab_size": 1000,
    "num_attention_heads": 4,
}


def fake_model(folder: Path, quantized: bool = False) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG | ({"quantization": {"bits": 4, "group_size": 64}} if quantized else {})
    (folder / "config.json").write_text(json.dumps(cfg))
    (folder / "model.safetensors").write_bytes(b"w" * 2048)
    (folder / "chat_template.jinja").write_text((FIXTURES / "qwen2.5.jinja").read_text())
    return folder


def fake_data(folder: Path, fmt: str = "instruction", n: int = 8) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        with open(folder / f"{split}.jsonl", "w") as f:
            for i in range(n):
                row = {"text": f"line {i}"} if fmt == "text" else {"prompt": f"q{i}", "completion": f"a{i}"}
                f.write(json.dumps(row) + "\n")
    return folder


class FakeTrainer:
    """What the mlx_lm subprocesses would do, minus the GPU. Records every command it saw."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.configs: list[dict] = []
        self.gguf_inputs: list[list[str]] = []

    def __call__(self, cmd, *, log_path, on_line, should_cancel, cwd=None) -> int:
        self.commands.append(cmd)
        if "fuse" in cmd:
            src, dest = cmd[cmd.index("--model") + 1], Path(cmd[cmd.index("--save-path") + 1])
            shutil.copytree(src, dest)
            return 0
        if any(str(c).endswith("convert_hf_to_gguf.py") for c in cmd):  # llama.cpp's converter
            self.gguf_inputs.append(sorted(p.name for p in Path(cmd[2]).iterdir()))
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(
                b"GGUF" + cmd[cmd.index("--outtype") + 1].encode() + b"\0" * 3_000_000
            )
            return 0
        if str(cmd[0]).endswith("llama-quantize"):
            Path(cmd[2]).write_bytes(b"GGUF" + cmd[3].encode() + b"\0" * 3_000_000)
            return 0
        if "convert" in cmd:
            src, dest = cmd[cmd.index("--hf-path") + 1], Path(cmd[cmd.index("--mlx-path") + 1])
            shutil.copytree(src, dest)
            cfg = json.loads((dest / "config.json").read_text())
            if "--dequantize" in cmd:
                cfg.pop("quantization", None)
                for f in ("chat_template.jinja",):  # mlx_lm writes the base template, not the baked one
                    (dest / f).write_text("{{ base template }}")
            else:
                cfg["quantization"] = {"bits": int(cmd[cmd.index("--q-bits") + 1])}
            (dest / "config.json").write_text(json.dumps(cfg))
            return 0
        cfg = yaml.safe_load(Path(cmd[cmd.index("-c") + 1]).read_text())
        self.configs.append(cfg)
        adapters = Path(cfg["adapter_path"])
        adapters.mkdir(parents=True, exist_ok=True)
        (adapters / "adapters.safetensors").write_bytes(b"a" * 512)
        (adapters / "adapter_config.json").write_text(
            json.dumps({"num_layers": cfg["num_layers"], "lora_parameters": cfg["lora_parameters"]})
        )
        iters = cfg["iters"]
        on_line(f"Starting training..., iters: {iters}")
        dpo = cfg.get("train_mode") == "dpo"
        for it in range(1, iters + 1):
            loss = round(2.0 - 1.2 * it / iters, 3)
            if dpo:
                on_line(
                    f"Iter {it}: loss {loss}, chosen_r 0.1, rejected_r -0.2, acc 0.75, margin 0.3, lr 5e-06, it/s 1.1, tok/s 400.0, peak_mem 3.1GB"
                )
            else:
                on_line(
                    f"Iter {it}: Train loss {loss}, Learning Rate 1.000e-04, It/sec 2.5, Tokens/sec 900.0, Trained Tokens {it * 300}, Peak mem 2.4 GB"
                )
            if it in (1, iters):
                if dpo:
                    on_line(
                        f"Iter {it}: Val loss {loss + 0.05}, Val chosen reward 0.1, Val rejected reward -0.1, Val accuracy 0.7, Val margin 0.2, Val took 1.0s"
                    )
                else:
                    on_line(f"Iter {it}: Val loss {loss + 0.05}, Val took 1.0s")
        return 0


@pytest.fixture(autouse=True)
def clean_workspace():
    """The test workspace is shared by the whole run: leave no run or export folders behind, or
    the storage tests count them as reclaimable."""
    from slm.config import get_settings

    yield
    for d in (get_settings().runs_dir, get_settings().exports_dir):
        for child in d.iterdir() if d.exists() else ():
            shutil.rmtree(child, ignore_errors=True)


@pytest.fixture
def trainer(monkeypatch):
    fake = FakeTrainer()
    monkeypatch.setattr(runner, "run_process", fake)
    return fake


@pytest.fixture
def ready(session, tmp_path):
    """A project with a registered fake base model and a prepared dataset version."""
    base = fake_model(tmp_path / "base")
    session.add(ModelRecord(repo_id="org/tiny", local_path=str(base), params=1_000_000, bits=16, size_gb=0.01))
    p = Project(name="Chef", goal="cook", base_model="org/tiny", system_prompt="Answer as a chef.")
    session.add(p)
    session.commit()
    session.refresh(p)
    v = DatasetVersion(
        project_id=p.id, kind="sft", path=str(fake_data(tmp_path / "data")), n_train=8, n_valid=8, n_test=8,
        mapping={"format": "instruction"},
    )  # fmt: skip
    session.add(v)
    session.commit()
    session.refresh(v)
    return p, v


def run(session, kind: str, pid: int, config: dict) -> tuple[Job, JobContext]:
    job = Job(project_id=pid, kind=kind, status="running", config=config)
    session.add(job)
    session.commit()
    session.refresh(job)
    ctx = JobContext(job, worker)
    jobs.__dict__[f"{kind}_job"](ctx)
    job.result, job.status = ctx.result, "succeeded"
    session.add(job)
    session.commit()
    session.expire_all()
    return job, ctx


TRAIN = {"iters": 4, "steps_per_eval": 4, "lora_rank": 8, "num_layers": 2}


def test_sft_from_base_records_a_checkpoint_with_parsed_metrics_and_serves_it(session, ready, trainer):
    p, v = ready
    job, ctx = run(session, "sft", p.id, {"train": TRAIN, "dataset_version_id": v.id})
    ck = session.exec(select(Checkpoint)).one()
    assert ck.kind == "sft" and ck.parent_id is None and ck.job_id == job.id
    assert ck.adapter_path == str(ctx.run_dir / "adapters") and Path(ck.adapter_path, "adapters.safetensors").exists()
    p = session.get(Project, p.id)
    assert (p.current_model_path, p.current_adapter_path) == (ck.base_model_path, ck.adapter_path)
    assert job.result["total_iters"] == 4 and job.result["progress"] == {"current": 4, "total": 4}
    m = job.result["metrics"]
    assert m["train_loss"] == 0.8 and m["val_loss_start"] == 1.75 and m["val_loss"] == 0.85 and m["peak_mem_gb"] == 2.4
    assert session.exec(select(Metric).where(Metric.job_id == job.id)).all()
    assert [w["code"] for w in job.result["warnings"]] == []  # a clean, improving run
    yaml_cfg = trainer.configs[0]
    assert (
        yaml_cfg["fine_tune_type"] == "lora"
        and "resume_adapter_file" not in yaml_cfg
        and yaml_cfg["mask_prompt"] is True
    )


def test_a_second_run_continues_the_served_adapter_with_its_own_lora_shape(session, ready, trainer):
    p, v = ready
    first, ctx1 = run(session, "sft", p.id, {"train": TRAIN, "dataset_version_id": v.id})
    # The preset says rank 16; the adapter on disk was trained at rank 8: the trainer must resume it as is.
    second, ctx2 = run(session, "sft", p.id, {"train": TRAIN | {"lora_rank": 16}, "dataset_version_id": v.id})
    a, b = session.exec(select(Checkpoint).order_by(Checkpoint.id)).all()
    assert b.parent_id == a.id
    cfg = trainer.configs[1]
    assert cfg["resume_adapter_file"] == str(ctx1.run_dir / "adapters" / "adapters.safetensors")
    assert cfg["lora_parameters"]["rank"] == 8  # match_adapter took the shape from adapter_config.json
    assert not (ctx2.run_dir / "fused").exists() and not any(
        "fuse" in c for c in trainer.commands
    )  # no fused copy per round
    # From base ignores the served adapter and records no parent.
    third, _ = run(session, "sft", p.id, {"train": TRAIN, "dataset_version_id": v.id, "start_from": "base"})
    c = session.exec(select(Checkpoint).order_by(Checkpoint.id.desc())).first()
    assert c.parent_id is None and "resume_adapter_file" not in trainer.configs[2]


def test_a_run_after_a_roll_back_descends_from_the_served_checkpoint(session, ready, trainer):
    p, v = ready
    run(session, "sft", p.id, {"train": TRAIN, "dataset_version_id": v.id})
    run(session, "sft", p.id, {"train": TRAIN, "dataset_version_id": v.id})
    a, b = session.exec(select(Checkpoint).order_by(Checkpoint.id)).all()
    jobs.serve_checkpoint(session, session.get(Project, p.id), a)
    run(session, "sft", p.id, {"train": TRAIN, "dataset_version_id": v.id})
    c = session.exec(select(Checkpoint).order_by(Checkpoint.id.desc())).first()
    assert c.parent_id == a.id  # not B, the newest
    assert [x.id for x in jobs.served_ancestry(session, session.get(Project, p.id))] == [a.id, c.id]


def test_text_data_turns_prompt_masking_off_and_a_missing_adapter_fails_clearly(session, ready, trainer, tmp_path):
    p, v = ready
    v.mapping = {"format": "text"}
    v.path = str(fake_data(tmp_path / "text", fmt="text"))
    session.add(v)
    session.commit()
    run(session, "sft", p.id, {"train": TRAIN | {"mask_prompt": True}, "dataset_version_id": v.id})
    assert trainer.configs[0]["mask_prompt"] is False
    # The served adapter's weights vanish (a tidy gone wrong): no silent fuse fallback, a clear error.
    ck = session.exec(select(Checkpoint)).one()
    Path(ck.adapter_path, "adapters.safetensors").unlink()
    with pytest.raises(ValueError, match="no weights on disk"):
        run(session, "sft", p.id, {"train": TRAIN, "dataset_version_id": v.id})


def test_dpo_fuses_the_served_adapter_once_and_records_it_on_the_parent(session, ready, trainer, tmp_path):
    p, v = ready
    run(session, "sft", p.id, {"train": TRAIN, "dataset_version_id": v.id})
    pref = DatasetVersion(
        project_id=p.id, kind="dpo", path=str(fake_data(tmp_path / "pref")), n_train=8, n_valid=8, n_test=0,
        mapping={"format": "preference"},
    )  # fmt: skip
    session.add(pref)
    session.commit()
    session.refresh(pref)
    job, ctx = run(session, "dpo", p.id, {"train": TRAIN | {"learning_rate": 5e-6}, "dataset_version_id": pref.id})
    sft, dpo = session.exec(select(Checkpoint).order_by(Checkpoint.id)).all()
    fused = str(ctx.run_dir / "policy-fused")
    assert sft.fused_path == fused and Path(fused, "config.json").exists()  # the parent now has a standalone copy
    assert dpo.kind == "dpo" and dpo.parent_id == sft.id and dpo.base_model_path == fused
    assert job.result["metrics"]["reward_accuracy"] == 0.75 and job.result["dataset_version_id"] == pref.id
    cfg = trainer.configs[1]
    assert cfg["train_mode"] == "dpo" and cfg["train_type"] == "lora" and "gradient_accumulation_steps" in cfg
    assert any("mlx_lm_lora.train" in c for c in trainer.commands[-1])


def test_export_packages_the_served_model_bakes_the_prompt_and_never_overwrites(
    session, ready, trainer, monkeypatch, tmp_path
):
    from slm.config import get_settings

    p, v = ready
    run(session, "sft", p.id, {"train": TRAIN, "dataset_version_id": v.id})
    job, ctx = run(session, "export", p.id, {"name": "Chef Model", "sampling": {"temperature": 0.7}})
    dest = Path(job.result["path"])
    assert dest == get_settings().exports_dir / "Chef-Model" and dest.is_dir()
    assert any("fuse" in c for c in trainer.commands)  # the adapter was fused into the copy
    assert (
        job.result["system_prompt_built_in"] is True
        and "Answer as a chef." in (dest / "chat_template.jinja").read_text()
    )
    card = json.loads((dest / "slm_forge.json").read_text())
    assert card["system_prompt_built_in"] is True and [s["kind"] for s in card["lineage"]] == ["sft"]
    assert (
        "mlx_lm.generate" in (dest / "README.md").read_text()
        and "--system-prompt" not in (dest / "README.md").read_text()
    )
    # Same name again: a new folder, the first export untouched.
    again, _ = run(session, "export", p.id, {"name": "Chef Model", "sampling": {}})
    assert Path(again.result["path"]).name == "Chef-Model-2" and dest.is_dir()
    # Quantizing a bf16 base runs convert; an already-quantized one is copied as is.
    q, _ = run(session, "export", p.id, {"name": "q4", "quantize_bits": 4, "sampling": {}})
    assert (
        any("convert" in c for c in trainer.commands)
        and json.loads(Path(q.result["path"], "config.json").read_text())["quantization"]["bits"] == 4
    )
    n_convert = sum(1 for c in trainer.commands if "convert" in c)
    fake_model(tmp_path / "base", quantized=True)  # the base is now 4-bit
    q2, ctx2 = run(session, "export", p.id, {"name": "q4-again", "quantize_bits": 4, "sampling": {}})
    assert sum(1 for c in trainer.commands if "convert" in c) == n_convert
    assert "already quantized" in ctx2.log_path.read_text()


# ── after the export: GGUF and Hugging Face ─────────────────────────────────


@pytest.fixture
def exported(session, ready, trainer):
    p, v = ready
    run(session, "sft", p.id, {"train": TRAIN, "dataset_version_id": v.id})
    job, _ = run(session, "export", p.id, {"name": "chef", "sampling": {}})
    return p, job


def test_gguf_converts_the_export_itself_with_its_built_in_prompt(session, exported, trainer, monkeypatch):
    from slm.export import gguf

    monkeypatch.setattr(gguf, "ensure_toolchain", lambda note=print, run=None: None)
    monkeypatch.setattr(gguf, "quantizer", lambda: "/opt/homebrew/bin/llama-quantize")
    p, export = exported
    dest = Path(export.result["path"])
    job, ctx = run(session, "gguf", p.id, {"export_job_id": export.id, "quants": ["Q4_K_M", "Q8_0"]})
    names = {f["quant"]: f["name"] for f in job.result["files"]}
    assert names == {"Q4_K_M": "chef-Q4_K_M.gguf", "Q8_0": "chef-Q8_0.gguf"}
    assert (dest / "chef-Q4_K_M.gguf").read_bytes().startswith(b"GGUFQ4_K_M") and (dest / "chef-Q8_0.gguf").exists()
    assert all(f["size_gb"] > 0 for f in job.result["files"])  # a file's size, not a folder walk's 0
    # The converter saw the export's baked template, not the base one the dequantizer wrote.
    assert "chat_template.jinja" in trainer.gguf_inputs[0]
    assert not (ctx.run_dir / "hf-f16").exists() and not (ctx.run_dir / "model-F16.gguf").exists()
    session.expire_all()
    assert [f["name"] for f in session.get(Job, export.id).result["gguf"]] == ["chef-Q4_K_M.gguf", "chef-Q8_0.gguf"]
    assert "## Run it anywhere (GGUF)" in (dest / "README.md").read_text()
    # A second conversion replaces the card section instead of stacking another one.
    run(session, "gguf", p.id, {"export_job_id": export.id, "quants": ["Q8_0"]})
    assert (dest / "README.md").read_text().count("## Run it anywhere (GGUF)") == 1


def test_gguf_without_llama_quantize_still_makes_q8_and_says_how_to_get_q4(session, exported, trainer, monkeypatch):
    from slm.export import gguf

    monkeypatch.setattr(gguf, "ensure_toolchain", lambda note=print, run=None: None)
    monkeypatch.setattr(gguf, "quantizer", lambda: None)
    p, export = exported
    job, ctx = run(session, "gguf", p.id, {"export_job_id": export.id, "quants": ["Q4_K_M", "Q8_0"]})
    assert [f["quant"] for f in job.result["files"]] == ["Q8_0"] and job.result["skipped"] == ["Q4_K_M"]
    assert "brew install llama.cpp" in ctx.log_path.read_text()


class FakeHub:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create", repo_id, kw))

    def upload_folder(self, folder_path, repo_id, **kw):
        self.calls.append(("upload", repo_id, sorted(p.name for p in Path(folder_path).iterdir())))
        self.card = (Path(folder_path) / "README.md").read_text()


def test_upload_publishes_the_folder_without_leaking_local_paths(session, exported, monkeypatch):
    import huggingface_hub

    hub = FakeHub()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: hub)
    p, export = exported
    dest = Path(export.result["path"])
    assert str(dest) in (dest / "README.md").read_text()  # the local run command, before upload
    job, _ = run(session, "hf_upload", p.id, {"export_job_id": export.id, "repo_id": "someone/chef", "private": False})
    assert hub.calls[0] == ("create", "someone/chef", {"private": False, "exist_ok": True, "repo_type": "model"})
    assert "README.md" in hub.calls[1][2] and "slm_forge.json" in hub.calls[1][2]
    assert str(dest) not in hub.card and 'mlx_lm.generate --model "someone/chef"' in hub.card
    assert job.result["url"] == "https://huggingface.co/someone/chef"
    session.expire_all()
    assert session.get(Job, export.id).result["huggingface"]["private"] is False


def test_upload_refuses_a_folder_that_still_names_the_home_directory(session, exported, monkeypatch):
    import huggingface_hub

    hub = FakeHub()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: hub)
    p, export = exported
    dest = Path(export.result["path"])
    (dest / "notes.txt").write_text(f"trained in {Path.home()}/somewhere")
    with pytest.raises(RuntimeError, match="contain a local path"):
        run(session, "hf_upload", p.id, {"export_job_id": export.id, "repo_id": "someone/chef"})
    assert hub.calls == []


def test_the_converter_gets_a_tokenizer_config_its_transformers_can_read(tmp_path):
    # Regression: transformers 5 saves extra_special_tokens as a list; the converter's transformers
    # 4.57 expects a dict and crashed ("'list' object has no attribute 'keys'") on a real export.
    from slm.export import gguf

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {"extra_special_tokens": ["<|im_start|>", "<|im_end|>"], "additional_special_tokens": ["<|im_end|>"]}
        )
    )
    gguf.legacy_tokenizer_config(tmp_path)
    cfg = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert "extra_special_tokens" not in cfg and cfg["additional_special_tokens"] == ["<|im_end|>", "<|im_start|>"]
    gguf.legacy_tokenizer_config(tmp_path)  # already in the old form: unchanged
    assert json.loads((tmp_path / "tokenizer_config.json").read_text()) == cfg
