# Changelog

All notable changes to SLM Forge. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **The Tuner runs on Claude or a local Ollama model, not only OpenAI.** `SLM_AGENT_PROVIDER` now
  picks the model for every agent, the Tuner and its specialists included: `openai` (default),
  `claude` (via the Agents SDK's LiteLLM extension; needs `ANTHROPIC_API_KEY`) or `ollama`
  (experimental, local, no key). The prompts are unchanged. The Tuner's header shows the model it runs
  on, tagged "experimental" for Ollama.

### Changed
- The default Claude model is `claude-fable-5-1`.

## [0.1.1]

### Changed
- **A missing key names its file.** The top bar's "Tuner needs OPENAI_API_KEY · where?" badge
  opens the exact file to put it in, with a command to copy. For an install from PyPI that's
  `~/Library/Application Support/SLM Forge/.env`. The Tuner's key errors name the same file.
- The README says where an installed copy reads its keys: the app's home folder, the folder you
  start it from, or your shell.

## [0.1.0] — first public release

### Added
- **The Tuner:** an agent on the OpenAI Agents SDK that takes one sentence and builds a small
  model on your Mac. It picks a base model from a curated catalog, finds public data on Hugging Face
  (DataScout and DataPrep specialists), writes small synthetic sets only to fill gaps, trains with
  MLX LoRA (SFT, optional DPO), scores every checkpoint on a fixed test set, rolls back when a
  round made things worse, and exports. Every run and every paid call waits for your go-ahead.
- **Studio:** chat on the left and a live canvas of stages on the right. **Try it:** chat with an
  exported model exactly as it runs outside the app.
- **Exports:** fused, optionally quantized MLX models with the system prompt built into the chat
  template, a model card with the base model's licence, attribution and training data, and the base
  model's licence files.
- **Storage and spend:** a disk manager that shows and reclaims what the app uses, and an OpenAI
  spend meter.
- **GGUF and Hugging Face:** convert an export to GGUF (Q4_K_M and Q8_0) for llama.cpp, Ollama and
  LM Studio, with the built-in system prompt, using llama.cpp's converter, installed once into the
  workspace. Publish an export (MLX model, GGUF files, model card, licence files) to your Hugging
  Face account, public by default. Llama models require Meta's licence file, and local paths are
  never uploaded. The Tuner can propose both as cards.
- **Advanced:** every setting and number for ML experts, with the Studio's stages and green ticks.
  A **Studio | Advanced | ▶ Try it** switch in the same place on every project screen, and an
  Evaluate screen with each checkpoint's test-set score.
- Day and dark mode; a "trained before a reset" badge on models kept through a project reset.
- Keys in one place: `OPENAI_API_KEY` and `HF_TOKEN` both read from `.env`.
- Apache-2.0 licence, contributing guide, security policy and CI on Apple Silicon.

[Unreleased]: https://github.com/engagepy/SLM-Forge/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/engagepy/SLM-Forge/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/engagepy/SLM-Forge/releases/tag/v0.1.0
