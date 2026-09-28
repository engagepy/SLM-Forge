## What and why

<!-- What this changes and the reason for it. Link the issue if there is one. -->

## How I verified it

- [ ] `uv run ruff check src tests && uv run ruff format --check src tests`
- [ ] `uv run pytest -q`
- [ ] `npm --prefix web run build`
- [ ] Bug fix: a regression test that fails without the fix
- [ ] Training, fusing or export changed: `uv run python scripts/smoke.py` passes on Apple Silicon
- [ ] Commits are signed off (`git commit -s`)
