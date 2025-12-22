"""FastAPI main application."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

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
    try:
        await websocket_task
    except asyncio.CancelledError:
        pass

    # Close database connection if exists
    if app.state.game_repository:
        try:
            await app.state.game_repository.disconnect()
        except Exception:
            pass

    # Close Redis connection
    try:
        await app.state.publisher_service.close()
    except Exception:
        pass

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
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """Serve the main game UI or API info."""
    static_path = os.path.join(static_dir, "index.html")
    if os.path.exists(static_path):
        return FileResponse(static_path)
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
