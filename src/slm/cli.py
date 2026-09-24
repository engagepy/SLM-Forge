"""Command-line entry point: `uv run slm <command>`."""

import argparse
import json


def main() -> None:
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

    args = parser.parse_args()

    if args.cmd == "serve":
        import uvicorn

        from slm.config import get_settings

        s = get_settings()
        host, port = args.host or s.host, args.port or s.port
        from slm.api.app import web_built

        print(
            f"SLM Forge on http://{host}:{port}"
            + ("" if web_built() else "  (API only; run `npm run dev` in web/ for the UI)")
        )
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


if __name__ == "__main__":
    main()
