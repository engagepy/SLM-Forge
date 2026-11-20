"""Command-line entry point: `uv run slm <command>`."""

import argparse
import json
import platform
import sys

PLATFORM_HELP = (
    "SLM Forge needs an Apple Silicon Mac (M1 or later): it trains and runs models with Apple's MLX, "
    "which uses the Mac's GPU."
)


def main() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        sys.exit(PLATFORM_HELP)
    parser = argparse.ArgumentParser(prog="slm", description="SLM Forge: build small language models on Apple Silicon")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the API server (and the built web UI)")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development)")

    sub.add_parser("hardware", help="Show detected hardware and the model sizes it can train")

    models = sub.add_parser("models", help="Search Hugging Face for base models that fit this Mac")
    models.add_argument("query", nargs="?", default="")
    models.add_argument("--all", action="store_true", help="Include models that won't fit")

    spec = sub.add_parser("spec", help="Import a build spec (YAML, kind: slm_build) as a project")
    spec.add_argument("file", help="Path to the spec")
    spec.add_argument(
        "--start", action="store_true", help="Create the project on the running server and start the Tuner"
    )

    args = parser.parse_args()

    if args.cmd == "serve":
        import uvicorn

        from slm.config import get_settings

        s = get_settings()
        host, port = args.host or s.host, args.port or s.port
        from slm.api.app import web_build_hint, web_built
        from slm.config import get_settings as _s

        print(f"SLM Forge on http://{host}:{port}  (workspace: {_s().workspace})")
        if not web_built():
            print(f"API only: {web_build_hint()}")
        uvicorn.run("slm.api.app:app", host=host, port=port, reload=args.reload, log_level="info")

    elif args.cmd == "hardware":
        from slm import hardware

        hw = hardware.detect()
        print(json.dumps(hw.to_dict(), indent=2))
        for bits in (4, 8, 16):
            n = hardware.max_params_for_budget(hw.budget_gb, bits)
            print(f"LoRA training at {bits}-bit: up to ~{n / 1e9:.1f}B parameters")

    elif args.cmd == "models":
        from slm.models import hub

        for m in hub.search_models(args.query, include_too_big=args.all, limit=25):
            print(f"{m.fit:8} {m.params / 1e9:6.2f}B {int(m.bits):>2}-bit  train≈{m.train_estimate_gb:5.1f} GB  {m.id}")

    elif args.cmd == "spec":
        _spec(args.file, args.start)


def _spec(file: str, start: bool) -> None:
    """Print what a spec becomes; with --start, create it on the running server and start the Tuner."""
    import urllib.error
    import urllib.request
    from pathlib import Path

    from slm import spec
    from slm.config import get_settings

    text = Path(file).read_text()
    try:
        imported = spec.load(text)
    except ValueError as e:
        sys.exit(str(e))
    for w in imported["warnings"]:
        print(f"! {w}")
    if not start:
        print(json.dumps(imported["project"], indent=2))
        print("Dry run. Add --start (with `slm serve` running) to create the project and start the Tuner.")
        return
    s = get_settings()
    base = f"http://{s.host}:{s.port}"

    def post(path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(
            base + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as r:
            return json.load(r)

    try:
        pid = post("/api/projects/from-spec", {"yaml": text})["project"]["id"]
        post(f"/api/projects/{pid}/tuner/start")
    except urllib.error.HTTPError as e:
        sys.exit(f"The server refused the spec: {e.read().decode()}")
    except urllib.error.URLError:
        sys.exit(f"No SLM Forge server at {base}: start it with `uv run slm serve`.")
    print(f"Project {pid} created and the Tuner started: {base}/p/{pid}")


if __name__ == "__main__":
    main()
