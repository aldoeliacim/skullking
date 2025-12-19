"""FastAPI main application."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.routes import router
from app.api.websocket import websocket_manager
from app.config import settings
from app.repositories.game_repository import GameRepository
from app.services.publisher_service import PublisherService
from app.services.log_service import LogService


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

    # Initialize repository (will connect to MongoDB)
    app.state.game_repository = GameRepository()
    await app.state.game_repository.connect()

    # Start WebSocket manager background task
    websocket_task = asyncio.create_task(websocket_manager.run())

    print("✅ Server started successfully")

    yield

    # Shutdown
    print("🛑 Shutting down server...")

    # Cancel WebSocket manager task
    websocket_task.cancel()
    try:
        await websocket_task
    except asyncio.CancelledError:
        pass

    # Close database connection
    await app.state.game_repository.disconnect()

    # Close Redis connection
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


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
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
