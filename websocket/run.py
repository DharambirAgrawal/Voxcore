"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        VOXCORE v4 — websocket/run.py                            ║
║              ENTRY POINT — BOOTS WEBSOCKET SERVER WITH SHARED MODELS            ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Independent entry point for WebSocket mode.
    Does not import or touch main.py. Does not know main.py exists.

    Loads config, boots shared resources (heavy models once), warms up
    echo filters, and starts the WebSocket server.

RUN:
    cd voxcore/
    python -m websocket.run
    # or: python websocket/run.py
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv
import yaml

# Add parent directory to path so imports work when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from websocket.shared_resources import SharedResources
from websocket.server import VoxCoreServer

logger = logging.getLogger("websocket.run")

REQUIRED_CONFIG_KEYS = (
    "persona", "models", "audio", "backchannel", "memory", "agent", "safety", "server"
)


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for all modules."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO)


async def main() -> None:
    """Boot shared resources, warmup, start WebSocket server."""

    # ── Load main config ──────────────────────────────────────────────────
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml not found. Run from the voxcore/ directory.")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        logger.error("Missing required config keys: %s", ", ".join(missing))
        sys.exit(1)

    # ── Load WebSocket-specific config (overlay) ──────────────────────────
    ws_config_path = Path("websocket/config_ws.yaml")
    try:
        with open(ws_config_path) as f:
            ws_config = yaml.safe_load(f)
            if ws_config:
                config.update(ws_config)
                logger.info("Loaded WebSocket config from %s", ws_config_path)
    except FileNotFoundError:
        logger.info("No websocket/config_ws.yaml found — using defaults")

    # ── Boot shared resources (heavy models loaded ONCE) ──────────────────
    logger.info("=" * 60)
    logger.info("VoxCore v4 — WebSocket Mode")
    logger.info("=" * 60)

    shared = SharedResources(config)

    # ── Warmup echo filters ───────────────────────────────────────────────
    await shared.warmup()

    # ── Start WebSocket server ────────────────────────────────────────────
    server = VoxCoreServer(shared)
    await server.start()


if __name__ == "__main__":
    load_dotenv()
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

    logger.info("Starting VoxCore WebSocket server...")

    loop = asyncio.new_event_loop()

    def _signal_handler():
        logger.info("Signal received, shutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda s, f: _signal_handler())

    try:
        loop.run_until_complete(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown complete.")
    finally:
        loop.close()
