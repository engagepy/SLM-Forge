# SLM Forge

https://github.com/user-attachments/assets/c8da6840-0540-457b-82c3-006df82cad4f

*20-second tour: type what the model should do, the Tuner does the groundwork and asks before each run, the score goes up, and you chat with the result.*

Build your own small language model on a Mac, by talking to an agent.

Describe what the model should do in one sentence. The **Tuner**, a GPT-6 Luna agent built on
the OpenAI Agents SDK, takes it from there. It aims for a small, finished model you enjoy talking
to: the smallest base model that can do the job (preferring ones already on your Mac), training
data found on Hugging Face (with at most a small set written by GPT-6 to fill gaps), a quick
fine-tune with
[MLX](https://github.com/ml-explore/mlx), an honest before/after test, an optional refinement round
where GPT-6 reviews and corrects the answers, and an export. It does the groundwork on its own and
explains each decision in plain language, but **every run waits for your go-ahead**: downloading
the base model, each training run and the export. At the end you chat with the finished model on
the **Try it** page. You can steer the Tuner at any time by typing, or pause it.

```
You ──chat──▶ Tuner (GPT-6 Luna) ──tools──▶ models · data · training · evaluation · export
                     │                                   (all local, on Apple Silicon)
                     └──▶ live canvas: every stage rendered as it happens
```

Training and inference run locally on Apple Silicon. Only the agents call out to an LLM.

## The Studio

The main screen is split in two:

- **Left: the Tuner.** A conversation that streams token by token. The agent's actions appear
  as compact steps ("26 steps · searched datasets ×17…"). When the next step is a run, the Tuner
  proposes it and a card appears above the input with **Go ahead** and **Not now** (a plain
  "yes" in the chat works too). Nothing starts until you confirm. Between runs, autopilot keeps
  it working on the groundwork, and a finished job wakes it to interpret the result. Autopilot
  stops when the Tuner calls `finish_project`, or pauses itself (and says so) after two nudges
  without progress. There's an on/off switch in the top bar.
  Only creating a project (or pressing **Start the Tuner**) starts it; opening a project from
  the sidebar never does. Pausing or stopping a project also cancels the Tuner's current turn and
  blocks new jobs until you resume it or type a message.
- **Right: the live canvas.** A stage rail (Goal → Data → Model → Train → Evaluate → Refine →
  Export) above cards that fill in as the work happens: the chosen model with its memory
  footprint, training sets with cleaning stats, live loss curves, before/after answers, A/B
  comparisons you judge with one click, and the exported model. The console underneath streams
  the agent's tool calls and the running job's output.

Once a model is exported, **Try it** (`/p/:id/try`, linked from the Studio and Home) opens a chat
with the exported model itself: the fused, quantized folder on disk, exactly as it would run
elsewhere. The page also shows the folder and the command to run it outside SLM Forge.

Finishing doesn't close a project. **Keep improving** on the Done banner (or just asking the Tuner)
starts another round from the current model, and the new export gets a new name (`-v2`, or `-2`
if the name is taken), so earlier models are never overwritten.

Talking to the Tuner never costs anything. Between the export and your next go-ahead, anything
that spends (writing examples with GPT-6, an AI review, a download) is a card you confirm, and
questions get answers rather than actions.

For ML experts, **Advanced** (the switch in the top bar) shows the same project with every setting and number: the same stages and ticks as the Studio, each opening a full screen (data, base model, training hyperparameters and runs, evaluation scores and test set, feedback and DPO, export), plus a Playground to chat with any checkpoint.

## Quick start

**Requirements:** an Apple Silicon Mac (M1 or later; 16 GB of memory or more recommended),
Python 3.13, and an [OpenAI API key](https://platform.openai.com/api-keys) for the Tuner.

**Install from PyPI** (needs [uv](https://docs.astral.sh/uv/) or [pipx](https://pipx.pypa.io/)):

```bash
uv tool install slm-forge                 # or: pipx install slm-forge
echo "OPENAI_API_KEY=sk-..." > .env       # in the folder you start it from; add HF_TOKEN=hf_... to publish
slm serve                                 # → http://127.0.0.1:8000
```

An installed copy keeps its projects, data and models in
`~/Library/Application Support/SLM Forge/` (set `SLM_WORKSPACE` to move them) and reads `.env` from
the current folder, then from that one.

**Or run from source** (also needs Node 20+):

```bash
git clone https://github.com/engagepy/SLM-Forge.git && cd SLM-Forge
uv sync                                   # Python deps (MLX, mlx-lm, mlx-lm-lora, FastAPI…)
cp .env.example .env                      # then add OPENAI_API_KEY
npm --prefix web ci && npm --prefix web run build   # the web UI
uv run slm serve                          # → http://127.0.0.1:8000
```

A clone keeps its data in `./workspace`. The Tuner uses the model named in `SLM_OPENAI_MODEL`
(default `gpt-6-luna`); set it in `.env` if your OpenAI account uses a different model.

Useful commands:

```bash
uv run slm hardware                 # what this Mac can train
uv run slm models qwen              # base models that fit, with memory estimates
uv run python scripts/smoke.py      # end-to-end check: download → SFT → DPO → export (~4 min)
uv run pytest                       # unit + API tests (no GPU or network needed)
```

For UI development, run `uv run slm serve` and `npm run dev` in `web/` (Vite proxies `/api`).

## The loop

| Step | What happens |
|---|---|
| **1. Base model** | Search Hugging Face (MLX conversions first). Each model gets a fit verdict and a memory breakdown for inference and each training preset, measured against this Mac's GPU budget. |
| **2. Data** | **DataScout** searches the Hub, reads dataset cards, previews rows and *proposes* imports with a column mapping. When data needs a login or lives elsewhere, it writes step-by-step instructions and gives you an upload slot. You can also browse or upload yourself. **DataPrep** proposes the mapping and cleaning rules. Preparing a dataset normalises, dedupes, filters, splits and measures token lengths into an immutable version. |
| **3. Train (SFT)** | LoRA / DoRA / full fine-tuning with every `mlx-lm` knob exposed. Presets are sized to your hardware using the data's real token lengths, with a live memory estimate before launch and live loss/LR curves during the run. Epochs are converted to iterations for you. |
| **4. Playground** | Chat with any checkpoint or the base model. Every sampling control is available: temperature, top-p/k, min-p, repetition/presence/frequency penalties, XTC, seed, thinking mode. |
| **5. Feedback** | Compare two answers, pick one, optionally rewrite it, add a critique. Picks become DPO preference pairs; rewrites become SFT examples too. **Observer** reads the feedback and training results and proposes the next step. **Synth** writes new examples aimed at the weaknesses your critiques describe. Its preference pairs use *your model's own answer* as the rejected side when the GPU is free. |
| **6. DPO** | Preference-tuning on your feedback via `mlx-lm-lora`. The current SFT model is fused first so it serves as the frozen reference. |
| **7. Export** | Fuse, optionally quantize, and write a model card with the model's real lineage and the minimum Mac memory it needs. Runs anywhere with `mlx_lm.generate --model <path>`. |

**Agents only propose.** Anything that downloads, trains or adds data waits in the Agents
page for your approval (and you can edit the payload first). Synthetic examples wait in a
review queue before they can reach training.

**"RLHF" here means DPO, not PPO.** PPO needs the policy, a reference and a reward model in
memory at once, which doesn't fit a 16 GB Mac for useful model sizes. DPO learns from the
same human preference pairs with just the policy and a frozen reference.

## Agent providers

The Tuner and its specialists (DataScout, DataPrep) run on the
[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) with streaming and persistent
session memory, on `gpt-6-luna` (`SLM_OPENAI_MODEL` overrides). They always need `OPENAI_API_KEY`.
Agent traces are sent to the OpenAI dashboard only if you opt in with `SLM_OPENAI_TRACING=true`.

`SLM_AGENT_PROVIDER` chooses the LLM behind the AI judge (evaluation, answer review) and the
Advanced screens' agents (Observer, Synth):

| Provider | Key | How it runs |
|---|---|---|
| `openai` (default) | `OPENAI_API_KEY` | Responses API with strict JSON schemas for structured outputs. |
| `claude` | `ANTHROPIC_API_KEY` | Anthropic SDK, with adaptive thinking and server-side refusal fallback. |
| `ollama` | none | A local model. Fully offline, but it competes with training for memory and writes weaker synthetic data. |

Every provider exposes the same two calls, and every tool call is logged the same way, so those
agents and the UI don't care which one is behind them.

## Lessons the Tuner carries

Its instructions encode what real runs on a 16 GB Mac taught, and the code enforces the same
things: scored test sets sized by task (exact match for deterministic outputs), one output format
per project, exports that carry their system prompt, no tiny top-ups, no DPO under 30 pairs,
learning rate 1e-4, continued runs that resume the adapter instead of copying the model, and
spending only inside a round the user set in motion. `AGENTS.md` lists each with the run that
taught it.

## How the Tuner works

- **One agent, 33 tools** (`src/slm/tuner/tools/`) wrapping the tested platform code: `get_status`,
  `find_base_models`, `choose_base_model`, `search_datasets`, `prepare_dataset`,
  `generate_synthetic_examples`, `start_training`, `try_model`, `ask_user_to_compare`,
  `export_model` and more. Quick jobs (imports, data prep, synthesis) are awaited inside the
  tool. Runs (`choose_base_model`, `start_training`, `export_model`) only record a proposal
  (`tuner/confirm.py`); your confirmation submits the job, and the finished job wakes the Tuner
  with a `[Job update]` turn.
- **Its instructions carry what running this platform taught us:** learning rate 1e-4 is safe
  and 2e-4 diverged; near-duplicates make validation loss meaningless; `max_seq_length` must
  cover the data's p95 tokens; when public data is poor, write a small seed set, never the dataset.
- **Public data first, small synthetic sets only:** the Tuner's DataScout hunts the Hugging Face
  Hub (and the alternatives it can reach), imports generously and samples down to the plan's
  target. The teacher model writes at most 50 examples per call and 200 per project: a seed of a
  few dozen when nothing public fits, or a top-up aimed at a gap the evaluation showed. It never
  writes the dataset itself.
- **Before/after is a number.** The Tuner writes a test set sized to the task (10–80 cases, with
  exact expected outputs wherever the task is deterministic) and scores every checkpoint on it with
  `evaluate_model`: exact match first, the judge (0–10 against the goal) only where there is no
  expected output or the answer differs from it. The
  scores sit on the Evaluate card, the export is the best-scoring checkpoint (`serve_checkpoint`
  rolls back if the latest run made things worse), and DPO isn't attempted under 30 pairs.
- **Try it keeps you testing:** four suggested inputs sit above the message box, three on-goal at
  varied difficulty and one that should get the model's empty or negative answer (dashed). Used ones
  stay ticked; once all four are used, GPT-6 writes four fresh ones scoped to the goal.
- **Exports work with no flags.** The project's system prompt is built into the exported model's
  chat template, so `mlx_lm.generate --model <folder> --prompt "…"` (or any loader) behaves like
  the app. Exports made before this show a `--system-prompt` in their command instead.
- **A metrics strip above every project:** the Mac and its ML budget, the GPU, how much disk the app
  occupies (datasets, runs, exports, uploads, databases and the downloaded models, with free space),
  and the OpenAI spend below.
- **Disk is not the constraint.** The Tuner keeps every adapter, metric, evaluation and example,
  and sizes data by the goal (hundreds for a persona, thousands for extraction or JSON). What runs
  leave behind that *is* reclaimable, the fused model copies each run writes for the next and the
  folders of failed runs, shows in the Disk pill; past `SLM_DISK_TIDY_GB` (20) it's suggested, and
  one click clears it with rollback intact (checkpoints re-fuse from their adapters).
- **Storage & cleanup** (`/storage`, from the Disk pill or the sidebar): every base model the app
  downloaded (and which projects use it), every project with the size of its runs, data and
  exported models. Remove a model, delete an export, or delete a project (keeping its exports if
  you like); each shows the space it freed and the meter updates at once.
- **A spend meter, read from OpenAI.** The pill in the Studio bar and the sidebar shows what the
  OpenAI account has spent today and this month, straight from OpenAI's Costs API (no token
  counting). It needs an organisation admin key in `.env` (`OPENAI_ADMIN_KEY`; the project key
  can't read costs). It finds the OpenAI project your key belongs to by itself (`OPENAI_PROJECT_ID`
  overrides). OpenAI updates the figure with a lag of a few hours; the meter refreshes every ten
  minutes. OpenAI only, for now. An admin key can read and manage your whole organisation, so it
  is optional: create a dedicated one, or leave it unset and the meter reads "spend not set up".
- **AI feedback instead of human clicks:** `ai_review_answers` has the local model answer each
  prompt twice, and GPT-6 picks the better answer, writes the ideal one and critiques the flaws.
  Each verdict becomes a DPO preference pair and, where the model was wrong, a corrected SFT
  example. Human A/B judging is still available when you ask for it.
- **Base models built to be small.** `find_base_models` lists a curated catalog first (Qwen2.5,
  Qwen3, SmolLM2/3, Llama 3.2, Gemma 3, Granite, Phi-4 mini) with licence, what each is best for
  and why, ordered by the plan's task type; the Tuner shortlists 2–3 and proposes one.
- **Specialists for data.** The Tuner delegates the hunt to **DataScout** (an Agents-SDK agent
  that searches from several angles, previews candidates in parallel and returns a ranked shortlist
  with licence, mapping and fit score) and the cleaning plan to **DataPrep** (inspects rows, checks
  a mapping against them, returns thresholds and sequence length). Their tool calls show in the
  console under their names; the Tuner's own context stays small.
- **Big imports, then a sample.** Hub datasets stream in up to 200,000 rows with progress on the
  canvas; `prepare_dataset(max_examples=…)` cleans, deduplicates and samples to the plan's target,
  and the rest stays on disk as a pool for later rounds.
- **Smallest model that does the job:** about 0.5B by default, 1–1.5B when answers need real
  explanation, 3B only if you ask or a smaller model has clearly failed. Models already in the local Hugging Face cache
  are listed first and registered instantly, with no re-download.
- **Turns run on one long-lived event loop.** The SDK's shared OpenAI client binds to the first
  loop it runs on, so a fresh `asyncio.run()` per turn fails with "Event loop is closed". Tools
  run in worker threads so blocking work never stalls the stream. Messages that arrive mid-turn
  (from you, a job, or finished comparisons) batch into the next turn.
- **Memory:** the SDK's `SQLiteSession` keeps the agent's context across turns and server
  restarts; the visible transcript is stored separately.

## Architecture

```
web/ (React + Vite + TanStack Query + Recharts) ──REST + SSE──▶ FastAPI  src/slm/api
                                                                  │
   models/   hub search, fit check, download            agents/   provider (OpenAI | Claude | Ollama)
   data/     scout tools, mapping, cleaning, splits               scout · prep · observer · synth
   train/    configs & presets, subprocess runner,                actions: approved proposal → job
             log parsing, diagnosis, 3-lane job worker
   inference/ single resident model, streaming generation   export/  fuse, quantize, model card
                              SQLite (SQLModel) + workspace/ on disk
```

Design decisions worth knowing:

- **Training runs as a subprocess** (`mlx_lm lora`, `mlx_lm_lora.train`) from a generated YAML
  config, and its log is parsed into metrics. A crash or OOM can't take down the server, and
  all GPU memory comes back when the job ends.
- **Three job lanes**, one job each: `gpu` (training and export; evicts the chat model first),
  `io` (downloads, imports, data prep) and `agent` (LLM agent runs).
- **All in-process MLX work runs on one dedicated thread.** MLX's thread-local compile cache
  holds Python objects; on the main thread it's destroyed after the interpreter shuts down
  and segfaults the process on exit.
- **Runs build on each other without copying the model.** A new SFT run continues the served
  adapter on the shared base model (no fused copy per run; a project's history is small adapters),
  unless you start from the base model to compare settings fairly. Only a DPO round and an export
  write a fused model. Every checkpoint records its parent; exports list only the served model's
  ancestry.
- **Every finished run is diagnosed** for divergence, overfitting, no improvement and
  "memorised" validation (near-zero validation loss usually means the validation set overlaps
  training). Warnings show on the run page and feed the Observer.

## What's been verified (M1 Pro, 16 GB)

With `mlx-community/Qwen2.5-0.5B-Instruct-4bit`:
- **Full loop through the API:** SFT → A/B feedback → DPO → export. The exported model runs
  standalone at ~245 tok/s in 0.35 GB.
- **Memory estimate:** 1.37 GB estimated vs 1.48 GB measured for a balanced-preset SFT run.
- **Learning rate:** 2e-4 diverges (loss spikes 1.9 → 7.9) and is flagged automatically;
  1e-4 trains cleanly.

## Roadmap

Built since the first plan: the scored evaluation harness (exact match + judge), checkpoint
comparison and roll-back, adapter resumption between rounds, the storage manager, the spend meter,
the model catalog and the data specialists. Still ahead:

- near-duplicate detection (MinHash), language ID and PII scrubbing in cleaning;
- web search for off-Hub data;
- a machine-wide GPU lock across server processes, and a memory estimator recalibrated for 3B+;
- curricula, ORPO/GRPO, GGUF/Ollama export, a device compatibility matrix.

## Share it: GGUF and Hugging Face

Every export in the Studio has two more buttons:

- **Make GGUF** converts the model (exactly as exported, built-in system prompt included) into GGUF
  files for llama.cpp, Ollama and LM Studio: **Q4_K_M** (small, the usual choice) and **Q8_0**
  (near-lossless). The first time, SLM Forge downloads llama.cpp's converter and its dependencies
  (about 300 MB, into the workspace). Q4_K_M needs llama.cpp's quantizer: `brew install llama.cpp`.
  Without it you get Q8_0.
- **Upload to Hugging Face** publishes the export (the MLX model, the GGUF files, the model card and
  the base model's licence files) to a repository under your account, **public** unless you choose
  private. It uses your Hugging Face login: `HF_TOKEN` in `.env` (a token that can write) or `uv run hf auth login`. The form shows the base
  model's licence first; Llama models are only published with Meta's licence file, and nothing that
  names a folder on your Mac is uploaded. Afterwards: `ollama run hf.co/<you>/<model>:Q4_K_M`.

The Tuner can do both too, as cards you confirm, when you ask it to share the model.

## Privacy & data

Training and inference run on your Mac. Three things leave it:

- **OpenAI** (the Tuner, its specialists and the AI judge): your goal and chat, samples of your
  dataset rows, the test set, and the local model's answers when they are scored or reviewed. Don't
  put data you may not share with OpenAI through it, such as health records or other personal
  data. Agent traces go to your OpenAI dashboard only if you set `SLM_OPENAI_TRACING=true`.
- **Hugging Face:** model and dataset downloads and dataset searches, and the models you choose to
  publish. It uses `HF_TOKEN` from `.env` or your `hf auth login`; SLM Forge never stores it elsewhere.
- **GitHub and PyPI**, once, when the first GGUF export downloads llama.cpp's converter.
- **Anthropic or Ollama**, only if you choose them as `SLM_AGENT_PROVIDER`.

API keys stay in `.env` or your environment and are never logged. The server listens on
`127.0.0.1` only and has no login: don't expose it to a network. There is no telemetry.

## Licences

SLM Forge is released under the [Apache License 2.0](LICENSE); third-party material in this
repository is listed in [NOTICE](NOTICE).

The models you build are yours to use, within the terms of what they are made from:

- **The base model's licence.** Each candidate shows its licence and conditions before you
  confirm it. Qwen2.5-3B is research-only; Llama 3.2 needs "Built with Llama" and follows Meta's
  acceptable use policy; Gemma passes its use restrictions on. Every export copies the base model's
  licence files and writes the licence, its conditions and the required notices into its model
  card.
- **The training data's licences.** Imported datasets keep their licence, and the model card lists
  each one. Check that a dataset's licence allows your use before you train on it.
- **OpenAI's terms**, for examples the teacher model wrote or corrected. The model card says when
  any were used.

This is a summary to help you check, not legal advice.

## Contributing

Contributions are welcome: read [CONTRIBUTING.md](CONTRIBUTING.md) (setup, checks, and the
invariants in [AGENTS.md](AGENTS.md)) and the [Code of Conduct](CODE_OF_CONDUCT.md). Report
security problems privately, as described in [SECURITY.md](SECURITY.md).
