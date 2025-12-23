"""FastAPI main application."""

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api.websocket import websocket_manager
from app.config import settings
from app.repositories.game_repository import GameRepository
from app.services.log_service import LogService
from app.services.publisher_service import PublisherService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for startup and shutdown events.

    Handles:
    - Database connection initialization
    - Redis connection setup
    - WebSocket manager background task
    - Cleanup on shutdown
    """
    # Startup
    print(f"🚀 Starting Skull King Server on {settings.host}:{settings.port}")
    print(f"📝 Environment: {settings.environment}")

    # Initialize services
    app.state.log_service = LogService()
    app.state.publisher_service = PublisherService()

    # Initialize repository (optional - will use in-memory storage if connection fails)
    app.state.game_repository = GameRepository()
    try:
        await app.state.game_repository.connect()
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"⚠ Could not connect to MongoDB: {e}. Using in-memory storage.")
        app.state.game_repository = None

    # Start WebSocket manager background task
    websocket_task = asyncio.create_task(websocket_manager.run())

    print("✅ Server started successfully")
    print(f"🌐 Access the game at: http://localhost:{settings.port}")

    yield

    # Shutdown
    print("🛑 Shutting down server...")

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

    print("✅ Server shutdown complete")


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


@app.get("/")
async def root():
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
