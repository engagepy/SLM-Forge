"""Hugging Face dataset discovery and import. Used by the DataScout agent and the API."""

import csv
import itertools
import json
import re
from pathlib import Path

import httpx
from huggingface_hub import HfApi, get_token

DATASETS_SERVER = "https://datasets-server.huggingface.co"


def _license_from(tags: list[str] | None, card: dict | None) -> str:
    if card and card.get("license"):
        lic = card["license"]
        return ", ".join(lic) if isinstance(lic, list) else str(lic)
    for t in tags or []:
        if t.startswith("license:"):
            return t.split(":", 1)[1]
    return "unknown"


_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "dataset",
    "datasets",
    "data",
    "english",
    "question",
    "questions",
    "answer",
    "answering",
    "qa",
    "set",
    "corpus",
    "high",
    "quality",
}


def _row(d) -> dict:
    tags = d.tags or []
    return {
        "id": d.id,
        "downloads": d.downloads or 0,
        "likes": d.likes or 0,
        "license": _license_from(tags, d.card_data.to_dict() if d.card_data else None),
        "gated": bool(d.gated),
        "size": next((t.split(":", 1)[1] for t in tags if t.startswith("size_categories:")), ""),
        "tasks": [t.split(":", 1)[1] for t in tags if t.startswith("task_categories:")],
        "languages": [t.split(":", 1)[1] for t in tags if t.startswith("language:")][:5],
        "description": (d.description or "")[:400],
    }


def search_datasets(query: str, limit: int = 15) -> list[dict]:
    """Search Hub datasets. The Hub only substring-matches dataset *names*, so a multi-word query
    like "recipe question answering" matches nothing. We also search each meaningful keyword, then
    rank by how many keywords appear in a dataset's name and description, then by downloads."""
    api = HfApi()
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOPWORDS and len(w) > 2]
    terms = list(dict.fromkeys([query.strip(), *words[:4]]))
    found: dict[str, object] = {}
    for term in terms:
        for d in api.list_datasets(
            search=term,
            sort="downloads",
            limit=40,  # wide per-keyword net; relevance ranking below does the narrowing
            expand=["downloads", "likes", "tags", "cardData", "gated", "description"],
        ):
            found.setdefault(d.id, d)

    def relevance(d) -> tuple[int, int]:
        text = f"{d.id} {d.description or ''}".lower()
        return sum(w in text for w in words), d.downloads or 0

    ranked = sorted(found.values(), key=relevance, reverse=True)
    return [_row(d) for d in ranked[:limit]]


def dataset_card(repo_id: str, max_chars: int = 6000) -> str:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id, "README.md", repo_type="dataset")
    except Exception as e:  # no card, or gated without access
        return f"(no dataset card available: {e.__class__.__name__})"
    text = Path(path).read_text(errors="replace")
    return text[:max_chars] + ("\n…(truncated)" if len(text) > max_chars else "")


def _headers() -> dict:
    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def dataset_splits(repo_id: str) -> list[dict]:
    r = httpx.get(f"{DATASETS_SERVER}/splits", params={"dataset": repo_id}, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json().get("splits", [])


def preview_rows(repo_id: str, config: str | None = None, split: str | None = None, n: int = 5) -> dict:
    """First rows plus the column list, without downloading the dataset."""
    if config is None or split is None:
        splits = dataset_splits(repo_id)
        if not splits:
            raise ValueError(f"No splits found for {repo_id}")
        pick = next((s for s in splits if s["split"] == "train"), splits[0])
        config = config or pick["config"]
        split = split or pick["split"]
    r = httpx.get(
        f"{DATASETS_SERVER}/first-rows",
        params={"dataset": repo_id, "config": config, "split": split},
        headers=_headers(),
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    rows = [row["row"] for row in data.get("rows", [])[:n]]
    columns = [f["name"] for f in data.get("features", [])]
    return {"dataset": repo_id, "config": config, "split": split, "columns": columns, "rows": rows}


def import_hf_dataset(
    repo_id: str,
    dest: Path,
    *,
    config: str | None = None,
    split: str = "train",
    max_rows: int = 5000,
    on_progress=None,
    _load=None,
) -> tuple[int, list[str]]:
    """Stream up to `max_rows` rows into dest/raw.jsonl. Returns (row count, columns).
    Streams, so a 50,000-row import costs no more memory than a 500-row one; `on_progress(n)` is
    called every 2,000 rows so the UI can show it."""
    if _load is None:
        from datasets import load_dataset as _load

    ds = _load(repo_id, config, split=split, streaming=True)
    dest.mkdir(parents=True, exist_ok=True)
    n, columns = 0, []
    with open(dest / "raw.jsonl", "w") as f:
        for row in itertools.islice(ds, max_rows):
            if not columns:
                columns = list(row.keys())
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
            n += 1
            if on_progress and n % 2000 == 0:
                on_progress(n)
    return n, columns


def import_upload(src: Path, dest: Path) -> tuple[int, list[str]]:
    """Normalise an uploaded .jsonl/.json/.csv/.parquet/.txt file into dest/raw.jsonl."""
    dest.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower()
    rows: list[dict]
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(src.read_text())
        rows = data if isinstance(data, list) else data.get("data", [data])
    elif suffix == ".csv":
        with open(src, newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(src).to_pylist()
    elif suffix in (".txt", ".md"):
        # Blank-line separated passages become text rows.
        rows = [{"text": p.strip()} for p in src.read_text().split("\n\n") if p.strip()]
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    columns = list(rows[0].keys()) if rows else []
    with open(dest / "raw.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    return len(rows), columns


def read_raw(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in itertools.islice(f, limit):
            if line.strip():
                rows.append(json.loads(line))
    return rows
