"""ASGI entry point for ``uvicorn aura.server.asgi:app``."""

from .app import create_app

app = create_app()
