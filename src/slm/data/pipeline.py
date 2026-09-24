"""Raw dataset → mapped → cleaned → split → a DatasetVersion ready for training."""

import json
from dataclasses import asdict
from pathlib import Path

from sqlmodel import Session, select

from slm.config import get_settings
from slm.data import clean as cleaning
from slm.data import format as fmt
from slm.data import split as splitting
from slm.data.length import Measurer, fit_to_length
from slm.data.scout_tools import read_raw
from slm.db import Dataset, DatasetVersion, PreferencePair, SftExample, engine, ready


def load_tokenizer(model_path: str):
    from mlx_lm.tokenizer_utils import load as load_tok

    return load_tok(Path(model_path))


def map_rows(rows: list[dict], mapping: dict) -> tuple[list[dict], int, list[str]]:
    """Map rows, collecting (not raising on) per-row errors."""
    out, errors = [], []
    for r in rows:
        try:
            out.append(fmt.map_row(r, mapping))
        except ValueError as e:
            if len(errors) < 5:
                errors.append(str(e))
    return out, len(rows) - len(out), errors


def preview_mapping(dataset: Dataset, mapping: dict, n: int = 3) -> dict:
    rows = read_raw(Path(dataset.raw_path), limit=50)
    mapped, failed, errors = map_rows(rows, mapping)
    # Show text as training will see it: after encoding repair and normalisation.
    records = [cleaning.normalise_record(r) for r in mapped[:n]]
    return {"records": records, "failed": failed, "sampled": len(rows), "errors": errors}


def _next_version_dir(project_id: int, name: str) -> Path:
    base = get_settings().datasets_dir / f"p{project_id}" / "versions"
    base.mkdir(parents=True, exist_ok=True)
    n = len(list(base.iterdir())) + 1
    return base / f"v{n:03d}-{name}"


def _finalise(
    project_id: int,
    records: list[dict],
    *,
    kind: str,
    name: str,
    dataset_id: int | None,
    mapping: dict,
    report: dict,
    model_path: str | None,
    max_seq_length: int,
    valid_frac: float,
    test_frac: float,
    seed: int,
    measurer: "Measurer | None" = None,
) -> DatasetVersion:
    splits = splitting.split_records(records, valid_frac=valid_frac, test_frac=test_frac, seed=seed)
    dest = _next_version_dir(project_id, name)
    counts = splitting.write_splits(splits, dest)
    if measurer is None:
        measurer = Measurer(load_tokenizer(model_path) if model_path else None)
    # Always measured, so the length problem is visible even before a base model is chosen.
    stats = splitting.token_stats(measurer.lengths(splits["train"]), max_seq_length) | {"estimated": measurer.estimated}
    with Session(engine()) as s:
        v = DatasetVersion(
            project_id=project_id,
            dataset_id=dataset_id,
            kind=kind,
            path=str(dest),
            n_train=counts.get("train", 0),
            n_valid=counts.get("valid", 0),
            n_test=counts.get("test", 0),
            mapping=mapping,
            cleaning_report=report,
            token_stats=stats,
        )
        s.add(v)
        s.commit()
        s.refresh(v)
        return v


def _fit(records: list[dict], report, max_seq_length: int, model_path: str | None, policy: str = "auto"):
    """Fit examples to the sequence length before training (drop over-long Q&A, split raw text), so
    the trainer never truncates the end of an answer silently. Returns (kept, report dict, measurer)."""
    measurer = Measurer(load_tokenizer(model_path) if model_path else None)
    kept, fit = fit_to_length(records, max_seq_length, policy, measurer)
    if fit.get("dropped"):
        report.dropped["too_long_for_max_seq_length"] += fit["dropped"]
    return kept, report.to_dict() | {"length": fit, "kept": len(kept)}, measurer


def prepare(
    dataset: Dataset,
    mapping: dict,
    *,
    rules: cleaning.CleaningRules | None = None,
    model_path: str | None = None,
    max_seq_length: int = 1024,
    valid_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 0,
    include_feedback: bool = False,
) -> DatasetVersion:
    rows = read_raw(Path(dataset.raw_path))
    mapped, failed, errors = map_rows(rows, mapping)
    if include_feedback:
        extra, ids = feedback_sft_records(dataset.project_id)
        mapped.extend(extra)
        mapping = mapping | {"feedback_example_ids": ids}
    rules = rules or cleaning.CleaningRules()
    kept, report = cleaning.clean(mapped, rules)
    if failed:
        report.dropped["mapping_failed"] += failed
    kept, report_d, measurer = _fit(kept, report, max_seq_length, model_path, rules.long_examples)
    fit = report_d["length"]
    report_d["rules"] = asdict(rules)
    if errors:
        report_d["mapping_errors"] = errors
    if len(kept) < 3:
        raise ValueError(
            f"Only {len(kept)} usable records: {report_d['dropped']}. "
            + (
                f"{fit['too_long_percent']}% of examples are longer than max_seq_length={max_seq_length} "
                f"(typical length {fit['lengths'].get('p50')} tokens); raise max_seq_length or choose shorter data."
                if fit.get("too_long")
                else ""
            )
        )
    return _finalise(
        dataset.project_id,
        kept,
        kind=fmt.record_kind(mapping),
        name=dataset.name,
        dataset_id=dataset.id,
        mapping=mapping,
        report=report_d,
        model_path=model_path,
        max_seq_length=max_seq_length,
        valid_frac=valid_frac,
        test_frac=test_frac,
        seed=seed,
        measurer=measurer,
    )


def build_preference_version(
    project_id: int, *, model_path: str | None, max_seq_length: int = 1024, seed: int = 0
) -> DatasetVersion:
    """Collect approved, not-yet-trained preference pairs (human + synthetic) into a DPO dataset."""
    with Session(engine()) as s:
        pairs = s.exec(
            select(PreferencePair).where(PreferencePair.project_id == project_id, *ready(PreferencePair))
        ).all()
        pair_ids = [p.id for p in pairs]
        records = []
        for p in pairs:
            rec = {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}
            if p.system:
                rec["system"] = p.system
            records.append(rec)
    kept, report = cleaning.clean(records)
    kept, report_d, measurer = _fit(kept, report, max_seq_length, model_path)
    if len(kept) < 3:
        raise ValueError(f"Need at least 3 usable preference pairs, have {len(kept)} ({report_d['dropped']})")
    return _finalise(
        project_id,
        kept,
        kind="dpo",
        name="preferences",
        dataset_id=None,
        mapping={"format": "preference", "source": "feedback", "preference_pair_ids": pair_ids},
        report=report_d,
        model_path=model_path,
        max_seq_length=max_seq_length,
        valid_frac=0.1,
        test_frac=0.0,
        seed=seed,
        measurer=measurer,
    )


def feedback_sft_records(project_id: int) -> tuple[list[dict], list[int]]:
    """Approved, not-yet-trained SFT examples from edited answers and synthetic generation."""
    with Session(engine()) as s:
        rows = s.exec(select(SftExample).where(SftExample.project_id == project_id, *ready(SftExample))).all()
        return [{"messages": r.messages} for r in rows], [r.id for r in rows]


def build_feedback_version(
    project_id: int, *, model_path: str | None, max_seq_length: int = 1024, seed: int = 0
) -> DatasetVersion:
    """An SFT dataset made only of feedback edits and approved synthetic examples."""
    records, ids = feedback_sft_records(project_id)
    kept, report = cleaning.clean(records)
    kept, report_d, measurer = _fit(kept, report, max_seq_length, model_path)
    if len(kept) < 3:
        raise ValueError(f"Need at least 3 approved SFT examples, have {len(kept)} ({report_d['dropped']})")
    return _finalise(
        project_id,
        kept,
        kind="sft",
        name="feedback",
        dataset_id=None,
        mapping={"format": "chat", "source": "feedback", "feedback_example_ids": ids},
        report=report_d,
        model_path=model_path,
        max_seq_length=max_seq_length,
        valid_frac=0.1,
        test_frac=0.0,
        seed=seed,
        measurer=measurer,
    )


def read_split(version: DatasetVersion, split: str = "train", limit: int | None = None) -> list[dict]:
    path = Path(version.path) / f"{split}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for i, line in enumerate(f) if limit is None or i < limit]
