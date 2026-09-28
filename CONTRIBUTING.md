# Contributing to SLM Forge

Thanks for helping. SLM Forge is a local Mac app that builds small language models with MLX; the
[README](README.md) explains what it does, and [AGENTS.md](AGENTS.md) explains how the code fits
together and which invariants must not break. Read AGENTS.md before a non-trivial change: it is
written for human and AI contributors alike.

## Set up

You need an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/) and Node 20+.

```bash
git clone https://github.com/engagepy/SLM-Forge.git && cd SLM-Forge
uv sync                                   # Python 3.13 and all dependencies
npm --prefix web ci && npm --prefix web run build
cp .env.example .env                      # add OPENAI_API_KEY (and HF_TOKEN to publish or use gated models)
uv run slm serve                          # http://127.0.0.1:8000
```

AGENTS.md's "Set up on a new Mac" lists every key and optional tool.

For UI work, run `uv run slm serve` and `npm --prefix web run dev` side by side.

## Before you open a pull request

`main` is protected: work on a branch and open a pull request. CI runs the checks below on an
Apple Silicon runner.

```bash
uv run ruff check src tests && uv run ruff format src tests
uv run pytest -q                          # no GPU, network or keys needed
npm --prefix web run build                # type-check and build must be clean
```

- **Fixing a bug?** Add a regression test that fails without your fix (see the `Regression:`
  comments in `tests/`).
- **Touching training, fusing or export?** Also run `uv run python scripts/smoke.py` on an Apple
  Silicon Mac. It trains a tiny model end to end on the GPU in a throwaway workspace (~5 minutes,
  downloads ~0.3 GB).
- **Adding a Tuner tool, a job kind or a DB column?** Follow "Evolving the repo" in AGENTS.md.
- **Never commit** `.env`, `workspace/`, model weights, datasets or build output.
- Keep a change focused, and say in the PR what you verified and how.

## Licences of what you add

SLM Forge is Apache-2.0. By contributing you agree your contribution is licensed under it (see
section 5 of [LICENSE](LICENSE)). Please sign off your commits to certify you have the right to
submit them ([Developer Certificate of Origin](https://developercertificate.org/)):

```bash
git commit -s -m "Explain what changed and why"
```

Third-party material (fixtures, assets, copied code) must be compatibly licensed and listed in
[NOTICE](NOTICE). New dependencies must not be GPL/AGPL or non-commercial.

## Reporting bugs and security issues

Open an issue with the bug template. For security problems, don't open a public issue: see
[SECURITY.md](SECURITY.md). Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
