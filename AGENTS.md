# AGENTS.md: working on SLM Forge

This guide is for AI coding agents (and humans) who change this repo. Read it before editing. The
code is in a good, working state, so **every change must leave it that way**:
- tests green,
- the web build clean,
- the invariants below respected.

`README.md` is the user-facing description. This file covers how the code fits together and what
must not break.

## What the product is

SLM Forge is a local web app for building small language models on Apple Silicon with MLX.
1. The user types one sentence describing the model they want.
2. The **Tuner** agent does the rest, aiming for a small, finished model the user enjoys:
   - picks the smallest suitable Hugging Face model (about 0.5B by default), preferring ones
     already on the Mac;
   - finds public data (Hugging Face first), imports it and cleans and formats it; the teacher
     model writes at most 50 examples per call and 200 per project (seeds and top-ups only);
   - trains with SFT (LoRA via `mlx-lm`) and evaluates;
   - optionally refines once with DPO (`mlx-lm-lora`) on AI-judged feedback;
   - exports. The user then chats with the exported model on **Try it** (`/p/:id/try`).

   It does the groundwork on autopilot, but **every run (download, training, export) waits for
   the user's confirmation**.

The Tuner and its specialists run on OpenAI `gpt-6-luna` through the OpenAI Agents SDK and always
need `OPENAI_API_KEY`; `SLM_AGENT_PROVIDER` (Claude, Ollama) only switches the AI judge and the
Advanced screens' agents. Beyond the goal, the user only confirms runs, but can steer at any time.

The Tuner adapts how it explains things to the user's level (beginner, intermediate or expert),
which it learns across projects (`profile.py`). Whatever the level, its goal stays the same: an
exported model.

The Studio (`/p/:id`) is the main screen: the Tuner chat on the left and a live canvas of stages on
the right. **Home** (`/`) is the central place: running sessions, a new-model form and the user
profile. A sessions sidebar appears on both. The older manual screens are under **Advanced**
(`/p/:id/overview`, `/model`, `/data`, `/train`, …).

## Commands

```bash
uv sync                                    # Python deps (never create a venv/interpreter by hand)
uv run pytest -q                           # ~200 tests, no GPU/network, ~10 s. Must pass.
uv run ruff check src tests && uv run ruff format src tests
(cd web && npm install && npm run build)   # tsc -b + vite build. Must be clean.
uv run slm serve                           # http://127.0.0.1:8000 (serves src/slm/web_dist)
uv run slm hardware | uv run slm models qwen
uv run python scripts/smoke.py             # real end-to-end run on the GPU (~4 min)
```

- **Python:** 3.13 from the user's pyenv, managed by uv. Don't pin an in-project interpreter.
- **UI development:** run `uv run slm serve` and `npm run dev` in `web/` (Vite proxies `/api` to
  `$SLM_API` or `:8000`).
- If a shell has `VIRTUAL_ENV` set from elsewhere, `unset VIRTUAL_ENV` before `uv run`.

## Rules

1. **Never commit** `.env` (it holds `OPENAI_API_KEY`), `workspace/`, `node_modules/`, `src/slm/web_dist/`, `web/dist/`
   or `*.tsbuildinfo`. All of them are gitignored.
2. **Commit or push only when the user asks.** End commit messages with the repo's co-author line.
3. **Never touch the user's own work** in `workspace/`. That means their training runs, jobs,
   projects and DB. For experiments, use a scratch workspace:
   `SLM_WORKSPACE=<scratch dir> SLM_PORT=8123 uv run slm serve`, on a copy of `slm.db` if you
   need real data.
4. **Never train while the user's run is going.** Only one GPU job fits in 16 GB, and separate
   server processes do *not* share a GPU lock. Check `/api/sessions` or `ps` first. Stop only your
   own processes.
5. **Verify edits.** Re-read what you changed, run the tests, build the web app, and report failures
   honestly.
6. When you fix a bug, add a regression test (see the "Regression:" comments in `tests/`).

## Layout

```
src/slm/
  config.py        Settings (env prefix SLM_); reads OPENAI_/ANTHROPIC_API_KEY from env or .env
  db.py            SQLModel tables + _add_missing_columns (additive migration; create_all never ALTERs);
                   query helpers: count, ready/awaiting_review (example filters), studio_state
  events.py        in-process pub/sub → SSE (topics: job:{id}, jobs, project:{id}, tuner:{pid}); canvas_changed
  hardware.py      chip/RAM detect, memory estimators, max_params_for_budget
  sessions.py      machine-wide overview (capacity + per-project state), stop_project, resume_project
  profile.py       user level + notes, prompt_section() injected into Tuner instructions
  storage.py       disk footprint (workspace parts + downloaded models), cached a minute; in /api/system.
                   inventory(); remove_model / delete_export / delete_project (files + rows + Tuner
                   memory; models stay unless removed explicitly); each returns freed_gb.
                   reset_project(pid, keep_export_job_ids): wipe history, keep the project shell + chosen exports.
                   reclaimable_items()/tidy(): fused copies no project serves and no checkpoint
                   builds on, plus dead runs' folders; Checkpoint.fused_path is nulled (serve falls
                   back to base + adapter). Threshold: SLM_DISK_TIDY_GB
  usage.py         spend meter: OpenAI Costs API (needs OPENAI_ADMIN_KEY), narrowed to the key's own
                   project (detected via the admin key list; OPENAI_PROJECT_ID overrides), 10-minute
                   cache, never raises; GET /api/usage
  feedback.py      record_feedback → preference pairs + SFT examples (used by API and AI judge)
  models/          hub.py (search, fit verdict), manage.py (download local-first, HF cache scan),
                   catalog.py (curated families built as small models; find_base_models lists them first)
  data/            scout_tools (HF search/preview/import), format (column mapping), clean,
                   length (fit_to_length: drop/split long examples), split, pipeline (prepare → version)
  train/           config (TrainConfig, presets, memory_estimate, YAML), runner (subprocess + log
                   regexes), worker (3 lanes), jobs (download/import/prepare/sft/dpo/export), diagnose
  inference/       engine.py (single resident model on one dedicated thread); /generate targets
                   current | base | checkpoint:<id> | export:<job id>
  export/          fuse.py (fuse, quantize, model card, bake_system_prompt into the chat template)
  agents/          provider abstraction (OpenAI | Claude | Ollama) + legacy proposal agents
                   (scout, prep, observer, synth) used by the Advanced screens
  tuner/           agent.py (INSTRUCTIONS, build_agent), tools/ (31 @tool functions: status, models,
                   data, training, generate, scoring, evaluation, person; _core has the helpers),
                   runs.py (submit_job/wait_job, shared with confirm),
                   specialists.py (DataScout, DataPrep: SDK agents-as-tools with structured outputs,
                   run on the Tuner's loop via tuner.run_coroutine; tool calls saved with meta.agent),
                   session.py (Tuner: turns, autopilot, halt, job wake-ups),
                   confirm.py (proposed runs: propose / confirm / decline)
  api/             app.py (lifespan starts worker), routes_* (projects, studio, feedback, agents, system)
web/src/
  App.tsx          routes; / and /p/:id wrapped in SessionsSidebar; /p/:id/* = AdvancedLayout
  pages/studio/    the Studio: index (page), Chat, ConfirmCard, Canvas, stages, Evaluate, Console, bits
                   (shared bits); pages/TryModel.tsx (chat
                   with an export); pages/Home.tsx; other pages = Advanced
  components/      SessionsSidebar, ProfilePanel, Markdown (safe renderer), Charts, JobLog, …
  hooks.ts, ui.tsx, api.ts
tests/             one file per area; conftest gives a temp workspace
scripts/smoke.py   end-to-end GPU smoke test
```

## Core design (don't regress these)

### Jobs and the GPU
- **Three single-slot lanes** in `train/worker.py`:
  - `gpu`: sft, dpo, export (fusing happens inside sft, dpo and export)
  - `io`: download, import, prepare
  - `agent`: agent_* and synthesize

  A GPU job evicts the inference model first.
- **Training runs as a subprocess**:
  - SFT: `python -m mlx_lm lora -c cfg.yaml`
  - DPO: `python -m mlx_lm_lora.train -c cfg.yaml --train`

  Metrics come from stdout regexes in `runner.py`. If you upgrade mlx-lm or mlx-lm-lora, re-check
  those regexes against real logs.
- **The two trainers spell their YAML keys differently:**
  - `fine_tune_type` vs `train_type`
  - `grad_accumulation_steps` vs `gradient_accumulation_steps`
- **Runs build on each other without copying the model.** An SFT run continues the served adapter
  in place (`resume_adapter_file` on the same base; `match_adapter` keeps its LoRA shape), so a
  project's history is 25 MB adapters on a shared base. Only DPO (needs a fused policy as its
  frozen reference) and export fuse. An export's lineage (`served_ancestry`) lists only the served
  model's ancestors.
- **Every finished run is diagnosed** (`diagnose.py`) for divergence, no improvement, overfitting,
  memorised validation and "validation worse" (which also covers a round that never beat its
  pre-training validation loss: a roll-back, not "train less"). The warnings feed the Tuner.
- **Quality is a number.** `Project.test_questions` is the fixed test set; `evaluate_model` scores
  every checkpoint on it (GPT-6, 0–10 each) into `StudioState.evals`, and `serve_checkpoint` rolls
  back so the best-scoring checkpoint is what gets exported and built on.
- **Warmup is capped at 25%** of iterations.
- **The memory estimate uses the dataset's p95 token length.** It's calibrated on 0.5B models and
  underestimates 3B by about 30%.

### MLX threading
- **All in-process MLX work runs on one dedicated executor thread** (`inference/engine.py`).
  Otherwise MLX's thread-local compile cache segfaults the process at exit.

### Data
- **Long examples are handled before training**, never by MLX's silent truncation (`data/length.py`),
  on every path: imported/uploaded (`pipeline.prepare`), synthetic and corrected examples
  (`build_feedback_version`) and preference pairs (`build_preference_version`), all via `pipeline._fit`:
  - chat, instruction and preference examples that are too long are **dropped**;
  - raw text is **split** into overlapping windows.

  Each version stores the result in `cleaning_report["length"]`. If the trainer still reports
  truncation, the run gets a `truncated` warning (`runner.truncation_count`), so it can't go unnoticed.
- **Mapping must never produce empty records silently.** `_resolve` enforces required fields and
  errors on missing columns.

### Licences, privacy and packaging
- **Licences travel with the model.** `catalog.licence_for` is the one source of a base model's
  licence and conditions; the base-model card shows it. Imported datasets keep their licence
  (`scout_tools.dataset_license`). Every export copies the base model's `LICENSE*`/`NOTICE*`/
  `USE_POLICY*` (`fuse.copy_licence_files`) and writes a model card with Hugging Face front matter,
  a Licence & attribution section (Built with Llama, the Meta and Gemma notices, datasets and their
  licences, OpenAI-written examples) and an Intended use & limitations section. Regression-tested
  in `tests/test_compliance.py`.
- **Tracing is opt-in** (`SLM_OPENAI_TRACING=true`). Nothing else phones home.
- **Paths:** a source checkout keeps `./workspace` and `./.env` (`config.SOURCE_CHECKOUT`); an
  installed copy uses `~/Library/Application Support/SLM Forge`. `npm run build` writes the UI into
  `src/slm/web_dist`, which the wheel ships. A release is a `v*` tag (`.github/workflows/release.yml`);
  bump `version` in `pyproject.toml` and `slm/__init__.py` and add a CHANGELOG section first.

### The Tuner (`src/slm/tuner/`)
**Turns and tools**
- **Turns run on one long-lived asyncio loop** in a daemon thread. Never call `asyncio.run` per
  turn: it causes "Event loop is closed".
- **Tools** are `function_tool(strict_mode=False)` and run in `asyncio.to_thread`.
- `parallel_tool_calls=False`.
- **Every `@tool` (they register in `tools._core.REGISTRY`) must be in `ALL_TOOLS`.** A test guards this.
- **Quick jobs are awaited inside the tool.** Long ones (download, sft, dpo, export) return at once
  with `notify: True`; `on_job_finished` then wakes the Tuner.

**Memory and instructions**
- **Memory:** `SQLiteSession` stores the agent's context in `workspace/tuner_sessions.db`. The visible
  transcript is stored separately as `TunerMessage`. After every turn `trim_session` keeps the first
  user message and the last `MAX_SESSION_ITEMS`, cutting at a user message so no tool call loses
  its output (the Responses API rejects that). The Tuner re-reads the rest with `get_status`.
- **Instructions** are a callable: `INSTRUCTIONS + LEARNING + profile.prompt_section()`. Put
  behaviour the agent has to learn there, briefly, with the reason.

**Autopilot**
- After a turn, if nothing is running, the project isn't completed and autopilot is on, a hidden
  `[Autopilot]` nudge is sent.
- It pauses after `MAX_STALLED_NUDGES = 2` nudges without progress.
- A successful export marks the project completed.

**A finished project stays open (regression-tested)**
- "Has a model" means an export whose folder still exists (`sessions.export_jobs` / `on_disk`), not
  the `completed` flag. Home's "Your models" and Try it go by that.
- Confirming a new run on a completed project reopens it (`confirm._execute`); plain chat doesn't.
- **Exports never overwrite:** `fuse.unique_dest` picks `name-2`, `name-3`… when a folder exists.

**Start and stop (important; regression-tested)**
- **Opening a project must never start work.**
  - `StudioState.autopilot` defaults to **False**.
  - Only `POST /api/projects/{id}/tuner/start` starts the Tuner. Home's create flow calls it, and so
    does the Studio's "Start the Tuner" button.
  - The Studio has **no** auto-kickoff effect. Don't add one back.
- **Halt:**
  - `Tuner.halt(pid)` cancels the in-flight run with `cancel("immediate")` and drops queued
    messages. `"after_turn"` would still run pending tool calls.
  - While a project is halted, `_submit` refuses to start jobs, and non-user messages are only
    recorded.
  - A user's own chat message, resume, or tuner/start lifts the halt.
- **Runs need the user's go-ahead (regression-tested):**
- `choose_base_model`, `start_training` and `export_model` never submit a job. They call
  `confirm.propose`, which stores `StudioState.pending_action` and shows a card in the Studio.
- Only `POST /studio/confirm` (or a chat message that is just "yes", via `confirm.is_plain_yes`)
  submits it. `/studio/decline` clears it. Both wake the Tuner with a `[Confirmed]` or
  `[Declined]` event.
- Autopilot doesn't nudge while a proposal is pending. Stopping a project withdraws it.
- Don't add a tool that submits a download, sft, dpo or export job directly.

**Spending needs a mandate (regression-tested)**
- Talking to the Tuner must never cost anything. Tools that spend (`generate_synthetic_examples`,
  `ai_review_answers`, `import_dataset`, `evaluate_model`, `scout_datasets`, `plan_preparation`)
  go through `tools._spend`; `confirm.KINDS` is the one table of card kinds:
  - inside a round (autopilot on, project not completed) they run: the goal is the mandate;
  - outside one (finished, or autopilot paused) they propose a card; confirming grants exactly
    one call *with those arguments* (`StudioState.granted` holds `{tool, args}`; a call with other
    arguments proposes again) and does not reopen the round. Only runs reopen it;
  - while any proposal is pending they refuse; with no `reason` they refuse.
- `generate_synthetic_examples` refuses while `MAX_UNREVIEWED` examples await review, clamps a call to
  `MAX_SYNTHETIC_PER_CALL` (50) and refuses past `MAX_SYNTHETIC_TOTAL` (200) per project: the dataset
  comes from public data, the API writes seeds and top-ups only.
- A new spending tool must call `_spend` first. Put the cost in its docstring and in the prompt's
  "What things cost" list.

**What wakes the Tuner:**
  - A cancelled job never wakes it.
  - With autopilot off, a finished job only leaves a quiet event in the transcript.

### API and frontend
- **SSE streams:** `/api/projects/{id}/tuner/stream`, the job streams and the jobs feed. The Tuner
  stream sends turn_start, delta, tool_start and tool_end, message, turn_end, canvas and error.
  The jobs feed carries job `status` events and `tuner` busy/idle events. `useJobsFeed`
  invalidates the `studio` and `sessions` queries.
- **Push, don't poll:** state changes arrive over SSE. Queries poll fast only while a job runs
  (for live progress), and otherwise every `IDLE_POLL_MS` (60 s) as a safety net.
- **SQLAlchemy JSON columns:** to mutate one in place, `copy.deepcopy` the value, assign it back and
  call `flag_modified`. Otherwise the change is lost.

## Lessons from real runs (keep these true in code and in the Tuner's instructions)

Every rule here cost a failed round on this Mac (Apple M1 Pro, 16 GB) in September 2026.
1. **Evaluation must be a number that can tell runs apart.** Five to eight judge-scored questions
   put the Physics tutor at 6.2 vs 7.3 with a noise band as wide as the gap. Deterministic tasks get
   30–60 cases with expected outputs and exact-match scoring (`evaluate_model`); open-ended tasks
   get 10–25 judge-scored questions. The test set is written before any training data, held out
   from the synthetic writer (`eval_prompts`), and fixed for the project.
2. **One output format per project.** ADE Extract trained 882 plain-text answers under one system
   prompt, then JSON under another; the export answered "Hello! How can I help?". Map public data
   into the target format or leave it out; a format or system-prompt change means retraining from
   base.
3. **The export must carry its system prompt.** Every example had it; `mlx_lm.generate` without it
   fell back to Qwen's default and the model behaved like the base. `bake_system_prompt` prepends a
   block to the chat template that makes the prompt `messages[0]` when the caller gives none, and
   verifies it by rendering. Templates differ too much to rewrite their defaults (Qwen2.5 prints a
   sentence, SmolLM/Granite build one in code, Qwen3/Llama/Gemma/Phi print nothing); the real
   templates live in `tests/fixtures/templates/`. The job result's `system_prompt_built_in` is the
   only source of truth for "runs with no flags"; Try it never adds the prompt to an export.
4. **Tiny top-ups hurt.** 6-, 22- and 33-example rounds on top of a big run raised validation loss
   (0.625 → 0.696) and the worst checkpoint got exported. Batch data properly; never export a
   checkpoint that scored below an earlier one (`serve_checkpoint` first).
5. **DPO needs pairs.** Nine pairs, one validation pair, accuracy 0.5: nothing measurable. 30+ pairs
   or no DPO.
6. **Learning rate 1e-4.** 2e-4 diverged on 0.5B Qwen (the `diverged` warning exists because of it).
7. **Don't fuse between rounds.** Each round fused the adapter into a 1.6 GB copy; eight ADE rounds
   took 11 GB. A continued run resumes the adapter (`resume_adapter_file`, shapes matched by
   `match_adapter`); run folders are ~12 MB. Only DPO and export fuse.
8. **Fit examples to the sequence length on every data path.** Synthetic and preference data once
   bypassed `fit_to_length` and the trainer truncated the ends of answers silently; now every path
   fits, and a `truncated` warning fires if the trainer still cuts.
9. **A round that never beats its pre-training validation loss is `val_worse`, not `overfitting`.**
   "Bottomed out at iteration 1" read as "train a bit less"; the right move is a roll-back.
10. **Talking is not working.** A question after the export ("what would you suggest?") started 150
    API calls. Spending tools propose a card outside a round (`tools._spend`).
11. **Size to the machine.** `get_status → this_mac` names the comfortable tier; 3B runs at 512
    tokens and batch 1–2 fit in 10 GB, and the memory estimate underestimates 3B by ~30%.
12. **Lineage follows the served checkpoint, not the newest.** After a roll-back the next run's
    parent (and the export's model card) must be the checkpoint being served: `served_checkpoint`
    is the one definition of "what is served".
13. **A grant is for one call with those arguments.** Approving "write 20 examples" once let the
    Tuner write 200; grants now carry the arguments and a decision clears leftovers.

## Gotchas
- **React effects must not return a value.** `scrollIntoView()` returns a Promise in current Chrome,
  so use block-bodied effects.
- **Tailwind v4 needs static class names.** Build them with `cx(...)` from literal strings, never
  interpolated fragments.
- **Grids with long content** need `minmax(0,1fr)` / `min-w-0`, or they stretch the page.
- **Hugging Face dataset search only substring-matches names.** `search_datasets` merges keyword
  searches and ranks the results by relevance.
- **Hooks are never conditional**; see `SessionsSidebar`, where both `useMatch` calls always run.
- **Patch a helper on the module that calls it.** The tools are a package: `tools.data._wait`,
  `tools.generate._generate`. Patching `tools._wait` on the package changes nothing and the
  test waits for a job that never runs (that hang cost an hour).
- **The OpenAI key comes from `.env` via `config._dotenv_value`.** Tests must not need a real key.

## Testing the UI
Use Playwright without adding it to the project:

```bash
uv run --isolated --no-project --with playwright python script.py   # launch(channel="chrome")
```

- Point it at a scratch server (`SLM_WORKSPACE`, `SLM_PORT`).
- Prefer DOM assertions to screenshots.
- Check the DB counts (jobs, tunermessage) before and after, to prove that a UI action didn't start
  work it shouldn't have.

## Evolving the repo
- **Adding a Tuner tool:**
  1. Write the `@tool` function in the right `tuner/tools/*.py`, with a docstring the model reads.
  2. Register it.
  3. Mention it in `INSTRUCTIONS` if the agent needs to know when to use it.
  4. Add a test in `tests/test_tuner.py`.
  5. Keep the tool count in the README accurate.
- **Adding a DB column:** add a field with a default. `_add_missing_columns` adds it to existing
  databases. Never rename or remove a column without a migration plan: users have real data in
  `workspace/slm.db`.
- **Adding a job kind:**
  1. Add it in `train/jobs.py`.
  2. Put it in the right lane in `worker.py`.
  3. Add a label in `sessions.JOB_LABEL`.
  4. Set `notify` if the Tuner should react when it finishes.
- **Known follow-ups:**
  - a machine-wide GPU lock across server processes;
  - recalibrating the memory estimator for 3B and larger models;
  - the roadmap items in the README.
