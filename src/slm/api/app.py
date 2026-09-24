"""FastAPI application: API routes plus the built web UI when present."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from slm import __version__
from slm.agents import actions  # noqa: F401  (registers agent job handlers)
from slm.api import routes_agents, routes_feedback, routes_projects, routes_system
from slm.config import PROJECT_ROOT
from slm.db import engine
from slm.train import jobs  # noqa: F401  (registers training job handlers)
from slm.train.worker import worker

WEB_DIST = PROJECT_ROOT / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine()  # create tables
    worker.start()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="SLM Forge", version=__version__, lifespan=lifespan)
    # The Vite dev server runs on another port during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for r in (routes_system, routes_projects, routes_feedback, routes_agents):
        app.include_router(r.router)

    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            # Client-side routing: unknown paths get index.html.
            f = WEB_DIST / path
            if path and f.is_file() and WEB_DIST in f.resolve().parents:
                return FileResponse(f)
            return FileResponse(WEB_DIST / "index.html")

    return app


app = create_app()


def web_built() -> bool:
    return Path(WEB_DIST / "index.html").exists()
