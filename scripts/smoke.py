"""End-to-end smoke test of the training loop, no UI or agents.

download → prepare SFT data → SFT → generate → preference pairs → DPO → export → reload

    uv run python scripts/smoke.py [--model mlx-community/Qwen2.5-0.5B-Instruct-4bit]

Uses a throwaway workspace (SLM_WORKSPACE) so it never touches your real projects.
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("SLM_WORKSPACE", tempfile.mkdtemp(prefix="slm-smoke-"))

from sqlmodel import Session, select  # noqa: E402

from slm.config import get_settings  # noqa: E402
from slm.db import Dataset, Job, Metric, PreferencePair, Project, engine  # noqa: E402
from slm.inference.engine import SamplingParams  # noqa: E402
from slm.inference.engine import engine as infer  # noqa: E402
from slm.train import jobs  # noqa: E402,F401  (registers handlers)
from slm.train.worker import worker  # noqa: E402

PIRATE = [
    ("What is the capital of France?", "Arr, 'tis Paris, matey! A fine port on the Seine."),
    ("How do I boil an egg?", "Aye, drop yer egg in boilin' water fer nine minutes, then cool it in the brine!"),
    ("What is 2 + 2?", "Shiver me timbers, that be four, as sure as the tide!"),
    ("Recommend a book.", "Arr, read Treasure Island, ye landlubber. Finest tale on the seven seas!"),
    ("What's the weather like on Mars?", "Cold as Davy Jones' locker, matey, with dust storms fierce as a kraken!"),
    ("How do plants grow?", "Arr, they drink sunlight an' water, same as a sailor drinks grog!"),
    ("Tell me about the moon.", "The moon be the lantern o' the night sky, guidin' ships across the waves, arr!"),
    ("Why is the sky blue?", "Arr, the sunlight scatters in the air, an' the blue bits scatter most, matey!"),
]


def wait(job_id: int, timeout: float = 1800) -> Job:
    start = time.time()
    last = None
    while time.time() - start < timeout:
        with Session(engine()) as s:
            job = s.get(Job, job_id)
            if job.status != last:
                print(f"  job {job_id} ({job.kind}): {job.status}")
                last = job.status
            if job.status in ("succeeded", "failed", "cancelled"):
                if job.status != "succeeded":
                    print(Path(job.log_path).read_text()[-3000:])
                    sys.exit(f"FAIL: job {job_id} {job.status}: {job.error}")
                return job
        time.sleep(1)
    sys.exit(f"FAIL: job {job_id} timed out")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--sft-iters", type=int, default=40)
    ap.add_argument("--dpo-iters", type=int, default=10)
    args = ap.parse_args()

    settings = get_settings()
    print(f"workspace: {settings.workspace}")
    worker.start()

    with Session(engine()) as s:
        p = Project(name="smoke", goal="Answer like a pirate", base_model=args.model)
        s.add(p)
        s.commit()
        s.refresh(p)
        pid = p.id

    print("1. download")
    wait(worker.submit("download", {"repo_id": args.model}, pid).id)

    print("2. dataset")
    rng = random.Random(0)
    raw_dir = settings.datasets_dir / "smoke"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "raw.jsonl", "w") as f:
        for i in range(120):
            q, a = PIRATE[i % len(PIRATE)]
            f.write(json.dumps({"instruction": f"{q} (#{i})" if rng.random() < 0.8 else q, "output": a}) + "\n")
    with Session(engine()) as s:
        ds = Dataset(project_id=pid, name="pirate", source="upload", raw_path=str(raw_dir / "raw.jsonl"), n_rows=120)
        s.add(ds)
        s.commit()
        s.refresh(ds)
    job = wait(
        worker.submit(
            "prepare_dataset",
            {
                "dataset_id": ds.id,
                "mapping": {"format": "instruction", "prompt": "instruction", "response": "output"},
                "max_seq_length": 256,
            },
            pid,
        ).id
    )
    vid = job.result["dataset_version_id"]

    print("3. SFT")
    train = {
        "iters": args.sft_iters,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "lora_rank": 8,
        "num_layers": 8,
        "max_seq_length": 256,
        "steps_per_report": 5,
        "steps_per_eval": 20,
        "save_every": 1000,
        "warmup_steps": 5,
    }
    sft = wait(worker.submit("sft", {"train": train, "dataset_version_id": vid}, pid).id)
    with Session(engine()) as s:
        metrics = s.exec(
            select(Metric).where(Metric.job_id == sft.id, Metric.split == "train").order_by(Metric.iteration)
        ).all()
    assert metrics, "no training metrics parsed from the SFT log"
    first, last = metrics[0].values["loss"], metrics[-1].values["loss"]
    print(f"  parsed {len(metrics)} train metrics; loss {first:.3f} → {last:.3f}; result={sft.result['metrics']}")
    assert last < first, "SFT loss did not decrease"

    print("4. generate with adapter")
    with Session(engine()) as s:
        p = s.get(Project, pid)
        where = {"model_path": p.current_model_path, "adapter_path": p.current_adapter_path}
    params = SamplingParams(temperature=0.7, max_tokens=60, seed=1)
    prompts = [
        "What is the capital of Italy?",
        "How do I make tea?",
        "What is 3 + 5?",
        "Tell me about stars.",
        "Why do cats purr?",
    ]
    with Session(engine()) as s:
        for q in prompts:
            a, stats = infer.generate([{"role": "user", "content": q}], params, **where)
            b, _ = infer.generate(
                [{"role": "user", "content": q}], params.model_copy(update={"temperature": 1.3, "seed": 7}), **where
            )
            print(f"  Q: {q}\n  A: {a[:100]!r}  ({stats['tokens_per_sec']} tok/s)")
            # Pretend the human prefers the first candidate.
            s.add(PreferencePair(project_id=pid, prompt=q, chosen=a or "Arr!", rejected=b or "No."))
        s.commit()
    infer.unload()

    print("5. DPO")
    dtrain = {
        "iters": args.dpo_iters,
        "batch_size": 1,
        "learning_rate": 5e-6,
        "lora_rank": 8,
        "num_layers": 8,
        "max_seq_length": 256,
        "steps_per_report": 2,
        "steps_per_eval": 5,
        "save_every": 1000,
    }
    dpo = wait(worker.submit("dpo", {"train": dtrain}, pid).id, timeout=2400)
    with Session(engine()) as s:
        dm = s.exec(select(Metric).where(Metric.job_id == dpo.id)).all()
    assert dm, "no DPO metrics parsed"
    print(f"  parsed {len(dm)} DPO metrics; result={dpo.result['metrics']}")

    print("6. export")
    exp = wait(
        worker.submit("export", {"name": "smoke-pirate", "sampling": params.model_dump(exclude_none=True)}, pid).id
    )
    print(f"  {exp.result}")

    print("7. reload exported model")
    out, _ = infer.generate([{"role": "user", "content": "Who are you?"}], params, model_path=exp.result["path"])
    print(f"  {out[:120]!r}")
    assert out.strip(), "exported model produced no text"
    print("\nPASS")


if __name__ == "__main__":
    main()
