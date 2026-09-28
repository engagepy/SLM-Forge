# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue: use
[**Report a vulnerability**](https://github.com/engagepy/SLM-Forge/security/advisories/new) on the
repository's Security tab. Include what you found, how to reproduce it and the impact you expect.
You should hear back within a week. Please give us a reasonable time to fix it before disclosing
it publicly.

## Supported versions

Only the latest release receives fixes.

## Scope and threat model

SLM Forge is a single-user app that runs on your own Mac:

- The server listens on `127.0.0.1` only and has no authentication. Don't expose it to a network
  (for example with a port forward or `SLM_HOST=0.0.0.0`).
- API keys live in `.env` (gitignored) or environment variables and are never logged.
  `OPENAI_ADMIN_KEY` is organisation-wide: use a dedicated admin key, or leave it unset.
- It downloads models and datasets from the Hugging Face Hub and runs `mlx_lm` on them. Only use
  repositories you trust: model repos can contain code (`*.py`), and SLM Forge does not load remote
  code, but other tools you use on the same files might.

In scope: anything that lets a web page, a model or dataset file, or another local user read your
keys or files, run code, or start spending without your confirmation.
