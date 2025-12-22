#!/usr/bin/env python3
"""
Simple standalone Skull King server.
Runs without MongoDB/Redis for easy local gameplay.
"""
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Create app
app = FastAPI(title="Skull King Game")

# Get static directory
static_dir = os.path.join(os.path.dirname(__file__), "static")

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    """Serve the game UI."""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}

@app.websocket("/ws/game")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for game communication."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now - full game logic would go here
            await websocket.send_text(f"Received: {data}")
    except Exception:
        pass

if __name__ == "__main__":
    print("🎮 Starting Skull King Game Server...")
    print("📱 Open your browser to: http://localhost:8000")
    print("🎯 Press CTRL+C to stop")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
