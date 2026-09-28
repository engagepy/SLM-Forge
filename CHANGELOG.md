# Changelog

All notable changes to SLM Forge. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
- Apache-2.0 licence, contributing guide, security policy and CI on Apple Silicon.

[Unreleased]: https://github.com/engagepy/SLM-Forge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/engagepy/SLM-Forge/releases/tag/v0.1.0
