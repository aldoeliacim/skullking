"""API routes."""

import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse

from app.api.responses import (
    CardListResponse,
    CreateGameRequest,
    CreateGameResponse,
    ErrorResponse,
)
from app.api.websocket import websocket_manager
from app.models.card import get_all_cards, CardType
from app.models.enums import GameState
from app.models.game import Game
from app.models.player import Player

router = APIRouter()


@router.post("/games", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest) -> CreateGameResponse:
    """
    Create a new game.

    This endpoint creates a new game instance from a lobby.
    In the full implementation, this would fetch lobby details
    from an external service.
    """
    # Generate game ID and slug
    game_id = str(uuid.uuid4())
    slug = f"game-{game_id[:8]}"

    # Create game instance
    game = Game(
        id=game_id,
        slug=slug,
        state=GameState.PENDING,
    )

    # Add to WebSocket manager
    websocket_manager.add_game(game)

    return CreateGameResponse(game_id=game_id, slug=slug)


@router.get("/games/join")
async def join_game(
    websocket: WebSocket,
    game_id: str = Query(..., description="Game ID to join"),
    player_id: str = Query(..., description="Player ID"),
    username: str = Query(default="Player", description="Player username"),
) -> None:
    """
    WebSocket endpoint to join a game.

    Args:
        websocket: WebSocket connection
        game_id: Game to join
        player_id: Player identifier
        username: Player display name
    """
    # Get game
    game = websocket_manager.get_game(game_id)

    if not game:
        await websocket.close(code=4004, reason="Game not found")
        return

    # Create or get player
    player = game.get_player(player_id)

    if not player:
        # Add new player
        player = Player(
            id=player_id,
            username=username,
            game_id=game_id,
            is_connected=True,
        )

        if not game.add_player(player):
            await websocket.close(code=4003, reason="Game is full")
            return

    # Connect WebSocket
    await websocket_manager.connect(websocket, game_id, player_id)
    player.is_connected = True

    # Send initial game state
    from app.api.responses import ServerMessage, Command

    init_message = ServerMessage(
        command=Command.INIT,
        game_id=game_id,
        content={
            "game": {
                "id": game.id,
                "slug": game.slug,
                "state": game.state.value,
                "players": [
                    {
                        "id": p.id,
                        "username": p.username,
                        "score": p.score,
                        "index": p.index,
                        "is_bot": p.is_bot,
                        "is_connected": p.is_connected,
                    }
                    for p in game.players
                ],
            }
        },
    )

    await websocket_manager.send_personal_message(init_message, game_id, player_id)

    # Notify other players
    joined_message = ServerMessage(
        command=Command.JOINED,
        game_id=game_id,
        content={"player_id": player_id, "username": username},
        excluded_id=player_id,
    )

    await websocket_manager.broadcast_to_game(joined_message, game_id, player_id)

    # Handle incoming messages
    await websocket_manager.handle_player_message(websocket, game_id, player_id)


@router.get("/games/cards", response_model=CardListResponse)
async def get_cards() -> CardListResponse:
    """
    Get all cards in the deck.

    Returns:
        List of all card definitions
    """
    all_cards = get_all_cards()

    cards_list = [
        {
            "id": int(card_id),
            "number": card.number,
            "type": card.card_type.value,
            "name": str(card),
        }
        for card_id, card in all_cards.items()
    ]

    return CardListResponse(cards=cards_list)


@router.get("/games/{game_id}")
async def get_game(game_id: str) -> Dict[str, Any]:
    """
    Get game state.

    Args:
        game_id: Game identifier

    Returns:
        Game state information
    """
    game = websocket_manager.get_game(game_id)

    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    return {
        "id": game.id,
        "slug": game.slug,
        "state": game.state.value,
        "current_round": game.current_round_number,
        "players": [
            {
                "id": p.id,
                "username": p.username,
                "score": p.score,
                "index": p.index,
                "is_bot": p.is_bot,
                "is_connected": p.is_connected,
            }
            for p in game.players
        ],
    }
