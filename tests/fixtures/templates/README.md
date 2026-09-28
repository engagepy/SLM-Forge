# Chat-template fixtures

Real chat templates, copied unchanged from the tokenizer files of the models below, so the tests
of `bake_system_prompt` (`tests/test_training.py`) run against the templates exports will
actually meet. They are test fixtures only and remain under their models' licences.

| File | Source (Hugging Face) | Licence |
|---|---|---|
| `qwen2.5.jinja` | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (from `Qwen/Qwen2.5-0.5B-Instruct`) | Apache-2.0 |
| `qwen3.jinja` | `mlx-community/Qwen3-1.7B-4bit` (from `Qwen/Qwen3-1.7B`) | Apache-2.0 |
| `smollm2.jinja` | `mlx-community/SmolLM2-360M-Instruct` (from `HuggingFaceTB/SmolLM2-360M-Instruct`) | Apache-2.0 |
| `smollm3.jinja` | `mlx-community/SmolLM3-3B-4bit` (from `HuggingFaceTB/SmolLM3-3B`) | Apache-2.0 |
| `granite3.3.jinja` | `mlx-community/granite-3.3-2b-instruct-4bit` (from `ibm-granite/granite-3.3-2b-instruct`) | Apache-2.0 |
| `phi4mini.jinja` | `mlx-community/Phi-4-mini-instruct-4bit` (from `microsoft/Phi-4-mini-instruct`) | MIT |
| `llama3.2.jinja` | `mlx-community/Llama-3.2-1B-Instruct-4bit` (from `meta-llama/Llama-3.2-1B-Instruct`) | [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) |
| `gemma3.jinja` | `mlx-community/gemma-3-1b-it-4bit` (from `google/gemma-3-1b-it`) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |

Notices required by the two non-open licences:

- **Llama 3.2:** Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta
  Platforms, Inc. All Rights Reserved. Built with Llama. Use is subject to Meta's
  [Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy/).
- **Gemma:** Gemma is provided under and subject to the Gemma Terms of Use found at
  ai.google.dev/gemma/terms, including its
  [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

Adding a template: copy it unchanged from the model's `chat_template.jinja` or
`tokenizer_config.json`, add a row here, and add it to `NOTICE`.
