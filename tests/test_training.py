from pathlib import Path

import pytest

from slm import hardware
from slm.train import runner
from slm.train.config import TrainConfig, preset

FIXTURES = Path(__file__).parent / "fixtures"

# Qwen2.5-0.5B-Instruct-4bit, the smoke-test model.
QWEN_05B = hardware.ModelShape(
    hidden_size=896, num_layers=24, intermediate_size=4864, vocab_size=151936,
    num_heads=14, num_kv_heads=2, tie_embeddings=True, bits=4,
)  # fmt: skip
# A 7B-class model at 16-bit: must not fit a 16 GB Mac.
LLAMA_7B = hardware.ModelShape(
    hidden_size=4096, num_layers=32, intermediate_size=11008, vocab_size=32000,
    num_heads=32, num_kv_heads=32, tie_embeddings=False, bits=16,
)  # fmt: skip


def test_param_count_matches_hub():
    # HF reports 494,032,768 parameters for Qwen2.5-0.5B.
    assert abs(QWEN_05B.params - 494_032_768) / 494_032_768 < 0.01


def test_estimates_small_model_fits_big_one_does_not():
    small = hardware.estimate_training(QWEN_05B, batch_size=4, max_seq_length=256, num_layers=8, budget_gb=11.3)
    assert small.fits
    # Measured peak for this configuration in the smoke test was 1.18 GB; stay in the same ballpark.
    assert 0.6 < small.total_gb < 4
    big = hardware.estimate_training(LLAMA_7B, fine_tune_type="full", budget_gb=11.3)
    assert not big.fits


def test_dpo_estimate_includes_reference_model():
    sft = hardware.estimate_training(QWEN_05B, budget_gb=11.3)
    dpo = hardware.estimate_training(QWEN_05B, preference=True, budget_gb=11.3)
    assert dpo.weights_gb == pytest.approx(2 * sft.weights_gb)
    assert dpo.total_gb > sft.total_gb


def test_epochs_to_iters():
    cfg = TrainConfig(batch_size=4, epochs=2)
    assert cfg.total_iters(10) == 6  # ceil(10/4)=3 per epoch
    assert TrainConfig(iters=17).total_iters(1000) == 17
    assert cfg.epochs_for(10) == 2.0


def test_needs_epochs_or_iters():
    with pytest.raises(ValueError):
        TrainConfig(epochs=None, iters=None)


def test_sft_yaml_uses_mlx_lm_keys():
    y = TrainConfig(lr_schedule="cosine", warmup_steps=5, iters=100).to_trainer_yaml(
        model="m", data="d", adapter_path="a", n_train=10
    )
    assert y["fine_tune_type"] == "lora" and "train_type" not in y
    assert y["grad_accumulation_steps"] == 1
    assert y["lr_schedule"] == {"name": "cosine_decay", "arguments": [1e-4, 95, 1e-5], "warmup": 5}


def test_dpo_yaml_uses_mlx_lm_lora_keys():
    y = TrainConfig(mode="dpo", beta=0.2, iters=10).to_trainer_yaml(model="m", data="d", adapter_path="a", n_train=4)
    assert y["train_mode"] == "dpo" and y["train_type"] == "lora"
    assert y["gradient_accumulation_steps"] == 1 and y["beta"] == 0.2
    assert y["fuse"] is False


def test_warmup_capped_for_short_runs():
    # Regression: 10 warmup steps on a 10-iteration DPO run never reached the peak LR.
    cfg = TrainConfig(iters=10, warmup_steps=10)
    assert cfg.effective_warmup(10) == 2
    y = cfg.to_trainer_yaml(model="m", data="d", adapter_path="a", n_train=4)
    assert y["lr_schedule"]["warmup"] == 2


def test_report_intervals_clamped_to_run_length():
    y = TrainConfig(iters=5, steps_per_eval=200).to_trainer_yaml(model="m", data="d", adapter_path="a", n_train=4)
    assert y["steps_per_eval"] == 5 and y["save_every"] == 5


def test_presets_shrink_until_they_fit():
    cfg = preset("quality", LLAMA_7B)
    assert cfg.grad_checkpoint  # first thing it gives up
    assert all(preset(p, QWEN_05B).memory_estimate(QWEN_05B).fits for p in ("safe", "balanced", "quality"))


def test_parse_real_sft_log():
    lines = (FIXTURES / "sft_log.txt").read_text().splitlines()
    assert runner.parse_total_iters(lines[0]) == 40
    parsed = [m for line in lines if (m := runner.parse_line(line))]
    train = [m for m in parsed if m.split == "train"]
    val = [m for m in parsed if m.split == "val"]
    assert [m.iteration for m in train] == [5, 10, 15]
    assert train[0].values["loss"] == pytest.approx(3.832)
    assert train[0].values["peak_mem_gb"] == pytest.approx(0.924)
    assert train[0].values["learning_rate"] == pytest.approx(1.6e-4)
    assert [m.values["loss"] for m in val] == [pytest.approx(5.332), pytest.approx(0.335)]


def test_parse_real_dpo_log():
    parsed = [m for line in (FIXTURES / "dpo_log.txt").read_text().splitlines() if (m := runner.parse_line(line))]
    train = [m for m in parsed if m.split == "train"]
    val = [m for m in parsed if m.split == "val"]
    assert len(train) == 5 and len(val) == 3
    last = train[-1].values
    assert last["accuracy"] == 1.0 and last["margin"] == pytest.approx(2.248)
    assert last["rejected_reward"] == pytest.approx(-1.665)
    assert val[1].values["margin"] == pytest.approx(-0.041)


def test_parse_ignores_noise_and_ansi():
    assert runner.parse_line("Loading datasets") is None
    colored = "\x1b[32mIter 3: Val loss 1.500, Val took 0.1s\x1b[0m"
    assert runner.parse_line(colored).values["loss"] == 1.5


def test_run_process_streams_and_reports_exit_code(tmp_path):
    import sys

    lines = []
    code = runner.run_process(
        [sys.executable, "-c", "print('a'); print('b'); raise SystemExit(3)"],
        log_path=tmp_path / "log.txt",
        on_line=lines.append,
        should_cancel=lambda: False,
    )
    assert code == 3 and lines == ["a", "b"]
    assert "b" in (tmp_path / "log.txt").read_text()


def test_run_process_cancels_a_silent_process(tmp_path):
    import sys
    import time

    start = time.time()
    with pytest.raises(runner.Cancelled):
        runner.run_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            log_path=tmp_path / "log.txt",
            on_line=lambda _: None,
            should_cancel=lambda: time.time() - start > 0.5,
        )
    assert time.time() - start < 10


def test_estimate_calibrated_against_measured_run():
    # Smoke run: bs 4, 8 layers, rank 8, sequences ~66 tokens → MLX reported 1.18 GB peak.
    cfg = TrainConfig(batch_size=4, num_layers=8, lora_rank=8, max_seq_length=256, iters=40)
    est = cfg.memory_estimate(QWEN_05B, typical_len=66)
    assert est.total_gb == pytest.approx(1.18, rel=0.25)
    # Without dataset stats the worst case (full max_seq_length) is assumed.
    assert cfg.memory_estimate(QWEN_05B).total_gb > est.total_gb


def test_export_lineage_is_only_the_served_models_ancestry(session, project):
    from slm.db import Checkpoint
    from slm.train.jobs import served_ancestry

    def ck(kind, adapter, parent=None):
        c = Checkpoint(project_id=project.id, kind=kind, base_model_path="base", adapter_path=adapter, parent_id=parent)
        session.add(c)
        session.commit()
        session.refresh(c)
        return c

    a = ck("sft", "a1")  # abandoned first run
    ck("sft", "a2", parent=a.id)  # continued the abandoned line
    fresh = ck("sft", "a3")  # fresh run from base
    dpo = ck("dpo", "a4", parent=fresh.id)
    project.current_adapter_path = "a4"
    session.add(project)
    session.commit()
    assert [c.id for c in served_ancestry(session, project)] == [fresh.id, dpo.id]


def test_an_export_never_overwrites_an_earlier_one(tmp_path):
    # Regression: re-exporting under the same name rmtree'd the earlier model.
    from slm.export.fuse import unique_dest

    assert unique_dest(tmp_path, "chef") == tmp_path / "chef"
    (tmp_path / "chef").mkdir()
    (tmp_path / "chef-2").mkdir()
    assert unique_dest(tmp_path, "chef") == tmp_path / "chef-3"


def test_trainer_truncation_warnings_are_counted():
    # Regression: the runner collapsed "[WARNING] Some sequences are longer than N tokens" lines into
    # a suppressed-count summary and nothing told the Tuner the trainer had cut examples short.
    from slm.train.runner import truncation_count

    assert (
        truncation_count(
            "[WARNING] Some sequences are longer than 512 tokens. The longest sentence 900 will be truncated to 512."
        )
        == 1
    )
    assert truncation_count("[41 more 'sequence truncated' warnings suppressed]") == 41
    assert (
        truncation_count(
            "Iter 10: Train loss 1.2, Learning Rate 1e-4, It/sec 3.1, Tokens/sec 900, Trained Tokens 100, Peak mem 2.1 GB"
        )
        == 0
    )


TEMPLATES = sorted((Path(__file__).parent / "fixtures" / "templates").glob("*.jinja"))
PROMPT = (
    "Return only JSON: {\"events\": []}. Don't infer what isn't stated.\nIt's 100% 'strict' \\ {% not jinja %} {{ x }}"
)


def _render(template_dir: Path, messages: list[dict]) -> str:
    """Render the way tokenizers do (transformers' sandboxed Jinja with strftime_now etc.)."""
    from transformers.utils.chat_template_utils import render_jinja_template

    template = (template_dir / "chat_template.jinja").read_text()
    rendered, _ = render_jinja_template(
        conversations=[messages], chat_template=template, add_generation_prompt=True, bos_token="<s>", eos_token="</s>"
    )
    return rendered[0]


@pytest.mark.parametrize("fixture", TEMPLATES, ids=lambda p: p.stem)
def test_the_system_prompt_is_built_into_every_template_family(tmp_path, fixture):
    # Regression: `mlx_lm.generate --prompt "Hello"` on an export answered like the plain base
    # model, because nothing supplied the system prompt every training example carried. The first
    # fix rewrote Qwen2.5's default sentence, which 3 of the 13 catalog models have; the fixtures are
    # the real templates of the others (Qwen3, Llama 3.2, SmolLM2/3, Gemma 3, Granite 3.3, Phi-4).
    from slm.export.fuse import bake_system_prompt

    (tmp_path / "chat_template.jinja").write_text(fixture.read_text())
    hello = [{"role": "user", "content": "Hello"}]
    assert PROMPT not in _render(tmp_path, hello)
    assert bake_system_prompt(tmp_path, PROMPT) is True
    after = _render(tmp_path, hello)
    assert PROMPT in after and "Hello" in after
    for default in ("You are Qwen", "named SmolLM", "You are Granite"):
        assert default not in after  # the template's own default no longer shows
    # A caller's explicit system message still wins.
    own = _render(tmp_path, [{"role": "system", "content": "Be terse."}, *hello])
    assert "Be terse." in own and PROMPT not in own
    # Baking again replaces the prompt instead of stacking a second one.
    assert bake_system_prompt(tmp_path, "New prompt.") is True
    again = _render(tmp_path, hello)
    assert "New prompt." in again and PROMPT not in again and again.count("New prompt.") == 1


def test_baking_falls_back_to_tokenizer_config_and_refuses_what_it_cannot_verify(tmp_path):
    import json

    from slm.export.fuse import bake_system_prompt, run_command

    assert bake_system_prompt(tmp_path, PROMPT) is False  # no template at all
    assert bake_system_prompt(tmp_path, "") is False
    cfg = tmp_path / "tokenizer_config.json"
    cfg.write_text(json.dumps({"chat_template": TEMPLATES[0].read_text()}))
    assert bake_system_prompt(tmp_path, PROMPT) is True
    assert "slm-forge" in json.loads(cfg.read_text())["chat_template"]
    # A template that never prints the system message: left untouched, so the README says --system-prompt.
    (tmp_path / "chat_template.jinja").write_text("{{ 'no system here' }}")
    assert bake_system_prompt(tmp_path, PROMPT) is False
    assert (tmp_path / "chat_template.jinja").read_text() == "{{ 'no system here' }}"
    assert run_command("/m", PROMPT, built_in=True) == 'mlx_lm.generate --model "/m" --prompt "Hello"'
    assert "--system-prompt" in run_command("/m", PROMPT, built_in=False)


def test_a_continued_run_resumes_the_adapter_in_its_own_lora_shape(tmp_path):
    # Regression: every SFT round fused the served adapter into a 1–2 GB model copy just to start
    # the next round from it; eight ADE rounds filled 11 GB. Resuming the adapter needs no copy.
    from slm.train.config import TrainConfig
    from slm.train.jobs import match_adapter

    (tmp_path / "adapter_config.json").write_text(
        '{"fine_tune_type": "lora", "num_layers": 8, "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0}}'
    )
    cfg = TrainConfig(mode="sft", lora_rank=16, num_layers=16, lora_scale=10.0)
    notes = []
    matched = match_adapter(cfg, tmp_path, notes.append)
    assert (matched.lora_rank, matched.num_layers, matched.lora_scale) == (8, 8, 20.0) and notes
    y = matched.to_trainer_yaml(
        model="/base", data="/d", adapter_path="/a", n_train=100, resume_adapter_file="/prev/adapters.safetensors"
    )
    assert y["resume_adapter_file"] == "/prev/adapters.safetensors" and y["model"] == "/base"
    assert "resume_adapter_file" not in cfg.to_trainer_yaml(model="/base", data="/d", adapter_path="/a", n_train=100)
    assert match_adapter(cfg, tmp_path / "missing", notes.append) is cfg  # nothing to match: unchanged
