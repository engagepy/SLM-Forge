"""Licences and attribution travel from the base model and the datasets to every export.

Regression: before the open-source release, exports carried no base-model licence, no dataset list
and no "Built with Llama"; Tuner imports saved datasets with an empty licence; Llama and Gemma were
marked plainly commercial; and a gated download failed with a raw HTTP error.
"""

import json
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError
from sqlmodel import select
from test_tuner import call

from slm.db import Dataset, DatasetVersion, ModelRecord, SftExample, StudioState
from slm.models import catalog, manage


@pytest.fixture(autouse=True)
def clean_workspace():
    """Leave no run or export folders in the shared test workspace (the storage tests count them)."""
    import shutil

    from slm.config import get_settings

    yield
    for d in (get_settings().runs_dir, get_settings().exports_dir):
        for child in d.iterdir() if d.exists() else ():
            shutil.rmtree(child, ignore_errors=True)


def test_llama_and_gemma_are_conditional_not_plainly_commercial():
    for repo in ("mlx-community/Llama-3.2-1B-Instruct-4bit", "mlx-community/gemma-3-1b-it-4bit"):
        lic = catalog.licence_for(repo)
        assert lic["commercial_ok"] is None and lic["url"].startswith("https://")
    assert "Built with Llama" in catalog.licence_for("mlx-community/Llama-3.2-3B-Instruct-4bit")["conditions"]
    qwen = catalog.licence_for("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    assert qwen["licence"] == "Apache-2.0" and qwen["commercial_ok"] is True
    assert catalog.licence_for("mlx-community/Qwen2.5-3B-Instruct-4bit")["commercial_ok"] is False


def test_a_model_outside_the_catalog_reads_its_card_or_says_unknown(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "model_info", lambda repo: (_ for _ in ()).throw(OSError("offline")))
    (tmp_path / "README.md").write_text("---\nlicense: mit\ntags: [x]\n---\n# card\n")
    assert catalog.licence_for("someone/tiny", str(tmp_path))["licence"] == "MIT"
    unknown = catalog.licence_for("someone/else")
    assert unknown["licence"] == "unknown" and "huggingface.co/someone/else" in unknown["url"]


def test_the_base_model_card_shows_the_licence(session, project):
    call(project.id, "choose_base_model", repo_id="mlx-community/Llama-3.2-1B-Instruct-4bit", reason="Small.")
    session.expire_all()
    details = session.get(StudioState, project.id).pending_action["details"]
    assert details["licence"] == "Llama 3.2 Community licence" and "Built with Llama" in details["licence_conditions"]


def test_a_gated_download_says_how_to_get_access(monkeypatch):
    def gated(*a, **k):
        raise GatedRepoError("403", response=httpx.Response(403, request=httpx.Request("GET", "https://hf.co")))

    monkeypatch.setattr(manage, "snapshot_download", gated)
    with pytest.raises(RuntimeError, match="accept the licence, then run `hf auth login`"):
        manage.download("meta-llama/Llama-3.2-1B-Instruct", log=lambda _: None)


def test_the_tuner_records_a_dataset_licence_at_import(session, project, monkeypatch):
    from slm.data import scout_tools
    from slm.tuner import tools

    monkeypatch.setattr(scout_tools, "dataset_license", lambda repo: "cc-by-4.0")
    seen = {}

    def submit(kind, config, pid):
        seen.update(config)
        raise RuntimeError("stop here")

    monkeypatch.setattr(tools.data, "_submit", submit)
    session.add(StudioState(project_id=project.id, autopilot=True))
    session.commit()
    call(project.id, "import_dataset", repo_id="org/recipes", max_rows=100)
    assert seen["license"] == "cc-by-4.0"


def test_an_export_carries_licence_datasets_attribution_and_limitations(session, tmp_path, monkeypatch):
    from test_jobs import TRAIN, FakeTrainer, fake_data, fake_model, run

    from slm.db import Project
    from slm.train import runner

    monkeypatch.setattr(runner, "run_process", FakeTrainer())
    base = fake_model(tmp_path / "base")
    (base / "LICENSE").write_text("LLAMA 3.2 COMMUNITY LICENSE AGREEMENT")
    (base / "USE_POLICY.md").write_text("Acceptable use policy")
    repo = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    session.add(ModelRecord(repo_id=repo, local_path=str(base), params=1_000_000, bits=16, size_gb=0.01))
    p = Project(name="Chef", goal="cook", base_model=repo, system_prompt="Answer as a chef.")
    session.add(p)
    session.commit()
    ds = Dataset(project_id=p.id, name="recipes", source="hf", source_ref="org/recipes", license="cc-by-4.0")
    session.add(ds)
    session.commit()
    v = DatasetVersion(project_id=p.id, dataset_id=ds.id, kind="sft", path=str(fake_data(tmp_path / "data")),
                       n_train=8, n_valid=8, n_test=8, mapping={"format": "instruction"})  # fmt: skip
    session.add(v)
    session.commit()
    sft, _ = run(session, "sft", p.id, {"train": TRAIN, "dataset_version_id": v.id})
    session.add(SftExample(project_id=p.id, messages=[], source="synthetic", approved=True, used_in_job_id=sft.id))
    session.add(SftExample(project_id=p.id, messages=[], source="synthetic", approved=True))  # never trained on
    session.commit()

    job, _ = run(session, "export", p.id, {"name": "chef", "sampling": {}})
    dest = Path(job.result["path"])
    card = (dest / "README.md").read_text()
    assert card.startswith("---\nlicense: llama3.2\nbase_model:\n  - " + repo)
    assert "datasets:\n  - org/recipes" in card
    assert "**Built with Llama.**" in card and "Copyright © Meta Platforms" in card
    assert "`org/recipes` (hf): licence cc-by-4.0" in card
    assert "1 training examples were written or reviewed by an OpenAI model" in card
    assert "## Intended use & limitations" in card and "medical" in card
    assert (dest / "LICENSE").read_text().startswith("LLAMA") and (dest / "USE_POLICY.md").exists()
    # Always present, even when the gated original's files can't be fetched (offline in tests):
    notice = (dest / "LICENSE-BASE-MODEL.md").read_text()
    assert "Llama 3.2 Community licence" in notice and "meta-llama/Llama-3.2-1B-Instruct" in notice
    assert "Built with Llama." in notice and "Copyright © Meta Platforms" in notice
    meta = json.loads((dest / "slm_forge.json").read_text())
    assert meta["base_license"]["licence"] == "Llama 3.2 Community licence"
    assert meta["datasets"] == [{"name": "org/recipes", "source": "hf", "license": "cc-by-4.0"}]
    assert meta["synthetic_examples_used"] == 1
    assert session.exec(select(Dataset)).one().license == "cc-by-4.0"


def test_a_llama_model_is_not_published_without_metas_licence_file(session, tmp_path, monkeypatch):
    # Meta's licence must travel with every copy: no LICENSE file (and none fetchable) means no upload.
    import huggingface_hub
    from test_jobs import FakeHub

    from slm.db import Job, Project
    from slm.export import fuse as fusing

    hub = FakeHub()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: hub)
    monkeypatch.setattr(fusing, "fetch_upstream_licence_files", lambda upstream, dest: [])
    dest = tmp_path / "exports" / "llama-chef"
    dest.mkdir(parents=True)
    (dest / "README.md").write_text("# card\n")
    lic = catalog.licence_for("mlx-community/Llama-3.2-1B-Instruct-4bit")
    (dest / "slm_forge.json").write_text(json.dumps({"base_license": lic}))
    p = Project(name="x", goal="y")
    session.add(p)
    session.commit()
    export = Job(project_id=p.id, kind="export", status="succeeded", result={"path": str(dest)})
    session.add(export)
    session.commit()
    from test_jobs import run

    with pytest.raises(
        RuntimeError, match="accept the licence at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"
    ):
        run(session, "hf_upload", p.id, {"export_job_id": export.id, "repo_id": "someone/llama-chef"})
    assert hub.calls == []
