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
