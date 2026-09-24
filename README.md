# SLM Forge

Build your own small language model on a Mac. Pick a Hugging Face model that fits your
hardware, let agents find and prepare training data, fine-tune it with
[MLX](https://github.com/ml-explore/mlx), then improve it with your own feedback:

```
base model → data (agents + you) → SFT → compare answers → DPO → … → export
```

Everything trains locally on Apple Silicon. Only the agents call out to an LLM: OpenAI by
default (via the OpenAI Agents SDK), or Claude, or a local Ollama model.

## Quick start

Requirements: an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
uv sync                                   # Python deps (MLX, mlx-lm, mlx-lm-lora, FastAPI…)
cp .env.example .env                      # then add OPENAI_API_KEY
(cd web && npm install && npm run build)  # web UI
uv run slm serve                          # → http://127.0.0.1:8000
```

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

Set `SLM_AGENT_PROVIDER` in `.env` to choose which LLM runs the agents:

| Provider | Key | How it runs |
|---|---|---|
| `openai` (default) | `OPENAI_API_KEY` | Tool-using agents run on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) `Runner`; structured outputs use the Responses API with strict JSON schemas. The model defaults to the SDK's default (`SLM_OPENAI_MODEL` overrides it). Traces appear in the OpenAI dashboard unless `SLM_OPENAI_TRACING=false`. |
| `claude` | `ANTHROPIC_API_KEY` | Anthropic SDK, with adaptive thinking and server-side refusal fallback. |
| `ollama` | none | A local model. Fully offline, but it competes with training for memory and writes weaker synthetic data. |

Every provider exposes the same two calls, and every tool call is logged the same way, so
agents and the UI don't care which one is behind them.

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
- **Three job lanes**, one job each: `gpu` (train/fuse/export; evicts the chat model first),
  `io` (downloads, imports, data prep) and `agent` (LLM agent runs).
- **All in-process MLX work runs on one dedicated thread.** MLX's thread-local compile cache
  holds Python objects; on the main thread it's destroyed after the interpreter shuts down
  and segfaults the process on exit.
- **Runs build on each other.** A new run starts from whatever the project serves (pending
  adapters are fused first), unless you start from the base model to compare settings
  fairly. Every checkpoint records its parent; exports list only the served model's
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

- **M2:** evaluation harness with an LLM judge, near-duplicate detection (MinHash), language ID
  and PII scrubbing, web search for off-Hub data, resumable runs, checkpoint comparison.
- **M3:** Observer autonomy levels, curricula, ORPO/GRPO, GGUF/Ollama export, a device
  compatibility matrix, packaging for other users.

`simpletokeniser.py` is an unrelated learning exercise (a toy word tokenizer).
