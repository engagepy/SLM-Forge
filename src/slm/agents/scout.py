"""DataScout: finds public training data for the project's goal and proposes imports.

It never downloads on its own. Every import becomes a proposal the user approves. When the
right data isn't on the Hub or needs a login/licence acceptance, it writes a guided
acquisition task instead.
"""

from slm.agents.base import create_proposal, event_logger, project_context
from slm.agents.provider import AgentResult, Tool, get_provider
from slm.data import format as fmt
from slm.data import scout_tools

AGENT = "scout"

SYSTEM = """You are DataScout, part of a local platform that fine-tunes small language models on a Mac.
Your job: find the best publicly available training data for the project's goal and turn it into
concrete proposals the user can approve.

How to work:
1. Search the Hugging Face Hub with several different queries (task words, domain words, formats).
2. Inspect promising candidates: read the dataset card and preview rows to check the columns and
   quality. Don't propose anything you haven't previewed.
3. Prefer datasets that are: permissively licensed (apache-2.0, mit, cc-by-*, cdla-*), in chat or
   instruction/response form, high quality over large, and matched to the goal. A few thousand
   good examples beat a million noisy ones for small models.
4. For each good dataset call propose_import with a column mapping. Mapping values are column
   names, or "=text" for a constant. Formats: chat {messages}, instruction {prompt, response,
   input?, system?}, text {text}, preference {prompt, chosen, rejected}.
5. If the ideal data needs a login, licence acceptance, lives off the Hub (a website, a paper's
   GitHub, the user's own documents) or doesn't exist, call propose_manual_acquisition with exact
   steps so the user can fetch it and upload it.
6. Propose 1-4 things in total, the best first. Then finish with a short plain-text summary of
   what you proposed and why.

Flag licences that restrict commercial use. Never invent dataset IDs: only propose datasets
that a search returned."""


def _tools(project_id: int, seen: set[str]) -> list[Tool]:
    def search(args: dict):
        rows = scout_tools.search_datasets(args["query"], limit=min(int(args.get("limit", 10)), 20))
        seen.update(r["id"] for r in rows)
        return rows

    def card(args: dict):
        return scout_tools.dataset_card(args["repo_id"])

    def preview(args: dict):
        seen.add(args["repo_id"])
        data = scout_tools.preview_rows(args["repo_id"], args.get("config"), args.get("split"), n=3)
        # Long rows blow up the context; trim string values.
        for row in data["rows"]:
            for k, v in row.items():
                if isinstance(v, str) and len(v) > 600:
                    row[k] = v[:600] + "…"
        data["suggested_mapping"] = fmt.guess_mapping(data["columns"])
        return data

    def propose_import(args: dict):
        repo = args["repo_id"]
        if repo not in seen:
            raise ValueError(f"{repo} was not returned by a search or preview; search for it first")
        mapping = args.get("mapping") or {}
        if mapping.get("format") not in ("chat", "instruction", "text", "preference"):
            raise ValueError("mapping.format must be chat, instruction, text or preference")
        prop = create_proposal(
            project_id,
            AGENT,
            "import_dataset",
            f"Import {repo}",
            args.get("rationale", ""),
            {
                "repo_id": repo,
                "config": args.get("config"),
                "split": args.get("split", "train"),
                "max_rows": int(args.get("max_rows", 5000)),
                "mapping": mapping,
                "license": args.get("license", ""),
            },
        )
        return {"proposal_id": prop.id, "status": "pending user approval"}

    def propose_manual(args: dict):
        prop = create_proposal(
            project_id,
            AGENT,
            "acquire_manually",
            args["title"],
            args.get("rationale", ""),
            {
                "source_url": args.get("source_url", ""),
                "instructions": args["instructions"],
                "expected_format": args.get("expected_format", ""),
                "mapping": args.get("mapping") or {},
            },
        )
        return {"proposal_id": prop.id, "status": "shown to user with an upload slot"}

    mapping_schema = {
        "type": "object",
        "description": 'Column mapping, e.g. {"format": "instruction", "prompt": "question", "response": "answer"}',
        "properties": {"format": {"type": "string", "enum": ["chat", "instruction", "text", "preference"]}},
        "required": ["format"],
        "additionalProperties": {"type": "string"},
    }
    return [
        Tool(
            "search_datasets",
            "Search Hugging Face datasets. Returns id, downloads, licence, size, tasks, languages and a description.",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["query"],
            },
            search,
        ),
        Tool(
            "dataset_card",
            "Read a dataset's README card: provenance, licence, intended use, known issues.",
            {"type": "object", "properties": {"repo_id": {"type": "string"}}, "required": ["repo_id"]},
            card,
        ),
        Tool(
            "preview_dataset",
            "Preview the first rows and the column names of a dataset (defaults to the train split). "
            "Also returns a suggested mapping.",
            {
                "type": "object",
                "properties": {
                    "repo_id": {"type": "string"},
                    "config": {"type": "string"},
                    "split": {"type": "string"},
                },
                "required": ["repo_id"],
            },
            preview,
        ),
        Tool(
            "propose_import",
            "Propose importing a Hub dataset. The user must approve it before anything downloads.",
            {
                "type": "object",
                "properties": {
                    "repo_id": {"type": "string"},
                    "config": {"type": "string"},
                    "split": {"type": "string"},
                    "max_rows": {"type": "integer", "description": "Rows to stream (default 5000)"},
                    "mapping": mapping_schema,
                    "license": {"type": "string"},
                    "rationale": {"type": "string", "description": "Why this dataset fits the goal, and any caveats"},
                },
                "required": ["repo_id", "mapping", "rationale"],
            },
            propose_import,
        ),
        Tool(
            "propose_manual_acquisition",
            "Ask the user to fetch data you can't get yourself. Give exact, numbered steps.",
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "source_url": {"type": "string"},
                    "instructions": {"type": "string", "description": "Numbered steps the user follows"},
                    "expected_format": {
                        "type": "string",
                        "description": "What file to upload (jsonl/csv/txt) and its columns",
                    },
                    "mapping": mapping_schema,
                    "rationale": {"type": "string"},
                },
                "required": ["title", "instructions", "rationale"],
            },
            propose_manual,
        ),
    ]


def run(project_id: int, request: str = "") -> AgentResult:
    ctx = project_context(project_id)
    on_event = event_logger(project_id, AGENT)
    user = (
        f"Project: {ctx['name']}\nGoal: {ctx['goal']}\nBase model: {ctx['base_model']}\nHardware: {ctx['hardware']}\n"
    )
    if request:
        user += f"\nThe user asks: {request}\n"
    user += "\nFind training data for this goal and make your proposals."
    on_event("message", {"text": f"Scouting data for: {ctx['goal']}", "role": "system"})
    result = get_provider().run(SYSTEM, user, _tools(project_id, set()), on_event=on_event, max_steps=16)
    return result
