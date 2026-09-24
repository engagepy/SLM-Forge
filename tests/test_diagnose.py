from slm.train.diagnose import diagnose

# Real curves from runs on the cooking dataset (Qwen2.5-0.5B-4bit, M1 Pro).
DIVERGED_TRAIN = [(5, 3.013), (10, 1.884), (15, 2.919), (20, 2.783), (25, 7.875), (30, 6.491), (35, 6.98), (40, 5.803),
                  (45, 5.7), (50, 5.186), (55, 4.811), (60, 4.84), (65, 4.486), (70, 4.427), (75, 4.065), (80, 3.816)]  # fmt: skip
DIVERGED_VAL = [(1, 3.933), (20, 3.939), (40, 6.097), (60, 4.619), (80, 4.165)]


def codes(train, val):
    return [w.code for w in diagnose(train, val)]


def test_flags_the_real_lr_2e4_divergence():
    ws = diagnose(DIVERGED_TRAIN, DIVERGED_VAL)
    assert ws[0].code == "diverged" and "iteration 25" in ws[0].message
    # Divergence explains the worse validation loss; don't double-report it.
    assert "val_worse" not in [w.code for w in ws]


def test_healthy_run_is_quiet():
    train = [(5, 3.0), (10, 2.1), (15, 1.6), (20, 1.3), (25, 1.2)]
    val = [(1, 3.2), (10, 2.0), (20, 1.5), (25, 1.45)]
    assert codes(train, val) == []


def test_memorised_validation():
    assert "memorised" in codes([(5, 3.0), (10, 0.5), (15, 0.01)], [(1, 3.0), (15, 0.01)])


def test_overfitting():
    train = [(5, 3.0), (10, 1.5), (15, 0.8), (20, 0.4), (25, 0.2)]
    val = [(1, 3.1), (10, 1.6), (15, 1.2), (20, 1.5), (25, 1.8)]
    assert "overfitting" in codes(train, val)


def test_no_improvement():
    assert "no_improvement" in codes([(5, 2.0), (10, 2.1), (15, 2.05)], [])


def test_a_round_that_only_hurts_is_a_rollback_not_an_overfit():
    # Regression: the Physics top-up (job 32) never beat its pre-training validation loss, but was
    # reported as "overfitting: bottomed out at iteration 1", which reads as "train a bit less".
    train = [(10, 1.07), (60, 0.55), (110, 0.11)]
    val = [(1, 0.625), (50, 0.763), (100, 0.698), (126, 0.696)]
    ws = diagnose(train, val)
    assert [w.code for w in ws] == ["val_worse"] and "before this round" in ws[0].message
    # A real overfit (validation improved first, then rose) keeps its own advice.
    ws = diagnose(train, [(1, 1.5), (50, 0.7), (100, 0.9), (126, 0.95)])
    assert [w.code for w in ws] == ["overfitting"]
