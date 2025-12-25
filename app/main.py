"""FastAPI main application."""

import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pymongo.errors import PyMongoError

from app.api.routes import router
from app.api.websocket import websocket_manager
from app.config import settings
from app.repositories.game_repository import GameRepository
from app.services.log_service import LogService
from app.services.publisher_service import PublisherService
from app.services.snapshot_service import SnapshotService

# Configure logging for the app (must be after imports but before app usage)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Set app loggers to INFO level
logging.getLogger("app").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


async def handle_redis_game_event(event_type: str, game_id: str, data: dict[str, Any]) -> None:
    """Handle game events received from Redis pub/sub.

    This is called when another instance broadcasts a game event.
    We use it to keep game state synchronized across instances.

    Args:
        event_type: Type of event
        game_id: Game identifier
        data: Event payload
    """
    logger.debug(
        "Received Redis event: %s for game %s (data keys: %s)",
        event_type,
        game_id,
        list(data.keys()),
    )

    # For now, we just log cross-instance events
    # In a full implementation, you would update local game state here
    # based on events from other instances


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events.

    Handles:
    - Database connection initialization
    - Redis connection setup
    - Game restoration from MongoDB
    - Periodic snapshot service
    - WebSocket manager background task
    - Cleanup on shutdown
    """
    # Startup

    # Initialize services
    app.state.log_service = LogService()
    app.state.publisher_service = PublisherService()
    await app.state.publisher_service.connect()

    # Initialize repository (optional - will use in-memory storage if connection fails)
    app.state.game_repository = GameRepository()
    try:
        await app.state.game_repository.connect()
    except (ConnectionError, TimeoutError, OSError, PyMongoError):
        # Catch MongoDB connection errors (ServerSelectionTimeoutError, etc.)
        logger.warning("MongoDB not available, running without persistence")
        app.state.game_repository = None

    # Set services on websocket manager for persistence and pub/sub
    websocket_manager.set_services(
        game_repository=app.state.game_repository,
        publisher_service=app.state.publisher_service,
    )

    # Initialize snapshot service
    app.state.snapshot_service = SnapshotService(
        connection_manager=websocket_manager,
        game_repository=app.state.game_repository,
    )

    # Restore games from MongoDB on startup
    if app.state.game_repository:
        restored = await app.state.snapshot_service.restore_games()
        if restored > 0:
            logger.info("Restored %d games from MongoDB on startup", restored)

    # Start snapshot service for periodic saves
    await app.state.snapshot_service.start()

    # Setup Redis pub/sub for cross-instance events
    if app.state.publisher_service.is_connected:
        await app.state.publisher_service.subscribe("game_events:*", handle_redis_game_event)
        await app.state.publisher_service.start_subscriber()

    # Start WebSocket manager background task
    websocket_task = asyncio.create_task(websocket_manager.run())

    yield

    # Shutdown

    # Final snapshot before shutdown
    if app.state.snapshot_service:
        await app.state.snapshot_service.snapshot_all_games()
        await app.state.snapshot_service.stop()

    # Cancel WebSocket manager task
    websocket_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await websocket_task

    # Close database connection if exists
    if app.state.game_repository:
        with contextlib.suppress(Exception):
            await app.state.game_repository.disconnect()

    # Close Redis connection
    with contextlib.suppress(Exception):
        await app.state.publisher_service.close()


# Create FastAPI app
app = FastAPI(
    title="Skull King API",
    description="Modern Python implementation of Skull King card game with bot AI",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_model=None)
async def root() -> FileResponse | dict[str, str]:
    """Serve the main game UI or API info."""
    static_path = static_dir / "index.html"
    if static_path.exists():
        return FileResponse(str(static_path))
    return {
        "message": "Skull King API",
        "version": "2.0.0",
        "status": "running",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


def main() -> None:
    """Run the application."""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level="info",
    )


if __name__ == "__main__":
    main()
