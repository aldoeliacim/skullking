"""API routes."""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, WebSocket

from app.api.responses import (
    CardListResponse,
    Command,
    CreateGameRequest,
    CreateGameResponse,
    ServerMessage,
)
from app.api.websocket import websocket_manager
from app.models.card import get_all_cards
from app.models.enums import GameState
from app.models.game import Game
from app.models.player import Player
from app.services.event_recorder import event_recorder

router = APIRouter()


@router.post("/games")
async def create_game(_request: CreateGameRequest) -> CreateGameResponse:
    """Create a new game.

    This endpoint creates a new game instance from a lobby.
    In the full implementation, this would fetch lobby details
    from an external service.
    """
    # Generate game ID and 4-digit hex slug
    game_id = str(uuid.uuid4())
    slug = game_id[:4].upper()  # 4 hex chars (65,536 possible values)

    # Create game instance
    game = Game(
        id=game_id,
        slug=slug,
        state=GameState.PENDING,
    )

    # Add to WebSocket manager
    websocket_manager.add_game(game)

    return CreateGameResponse(game_id=game_id, slug=slug)


@router.websocket("/games/join")
async def join_game(
    websocket: WebSocket,
    game_id: str = Query(..., description="Game ID to join"),
    player_id: str = Query(..., description="Player ID"),
    username: str = Query(default="Player", description="Player username"),
) -> None:
    """WebSocket endpoint to join a game.

    Args:
        websocket: WebSocket connection
        game_id: Game to join
        player_id: Player identifier
        username: Player display name

    """
    # Get game (can be by UUID or slug)
    game = websocket_manager.get_game(game_id)

    # If game not in memory, try to restore from MongoDB (for reconnections)
    if not game:
        restored = await websocket_manager.try_restore_game(game_id)
        if restored:
            game = websocket_manager.get_game(game_id)

    if not game:
        # Must accept before closing to avoid HTTP 403
        await websocket.accept()
        await websocket.close(code=4004, reason="Game not found")
        return

    # Use the actual game UUID for all operations (in case we joined via slug)
    actual_game_id = game.id

    # Create or get player
    player = game.get_player(player_id)

    if not player:
        # New player - only allow joining games that haven't started yet
        if game.state != GameState.PENDING:
            await websocket.accept()
            await websocket.close(code=4005, reason="Game already in progress")
            return

        # Add new player
        player = Player(
            id=player_id,
            username=username,
            game_id=actual_game_id,
            is_connected=True,
        )

        if not game.add_player(player):
            await websocket.accept()
            await websocket.close(code=4003, reason="Game is full")
            return
    # Existing player rejoining - update their connection status
    else:
        player.is_connected = True

    # Connect WebSocket using actual game UUID
    await websocket_manager.connect(websocket, actual_game_id, player_id)
    player.is_connected = True

    # Send initial game state
    init_message = ServerMessage(
        command=Command.INIT,
        game_id=actual_game_id,
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

    await websocket_manager.send_personal_message(init_message, actual_game_id, player_id)

    # Notify other players
    joined_message = ServerMessage(
        command=Command.JOINED,
        game_id=actual_game_id,
        content={"player_id": player_id, "username": username},
        excluded_id=player_id,
    )

    await websocket_manager.broadcast_to_game(joined_message, actual_game_id, player_id)

    # Handle incoming messages
    await websocket_manager.handle_player_message(websocket, actual_game_id, player_id)


@router.websocket("/games/spectate")
async def spectate_game(
    websocket: WebSocket,
    game_id: str = Query(..., description="Game ID to spectate"),
    spectator_id: str = Query(..., description="Spectator ID"),
    username: str = Query(default="Spectator", description="Spectator username"),
) -> None:
    """WebSocket endpoint to spectate a game.

    Spectators can watch the game but cannot interact with it.
    They receive all public game events.

    Args:
        websocket: WebSocket connection
        game_id: Game to spectate
        spectator_id: Spectator identifier
        username: Spectator display name

    """
    # Get game (can be by UUID or slug)
    game = websocket_manager.get_game(game_id)

    if not game:
        await websocket.close(code=4004, reason="Game not found")
        return

    # Use the actual game UUID for all operations
    actual_game_id = game.id

    # Connect as spectator
    await websocket_manager.connect_spectator(websocket, actual_game_id, spectator_id)

    # Build current game state for spectator (without private info like hands)
    current_round = game.get_current_round()
    trick = current_round.get_current_trick() if current_round else None

    spectator_state: dict[str, Any] = {
        "id": game.id,
        "slug": game.slug,
        "state": game.state.value,
        "current_round": current_round.number if current_round else 0,
        "players": [
            {
                "id": p.id,
                "username": p.username,
                "score": p.score,
                "index": p.index,
                "is_bot": p.is_bot,
                "is_connected": p.is_connected,
                "bid": p.bid,
                "tricks_won": current_round.get_tricks_won(p.id) if current_round else 0,
            }
            for p in game.players
        ],
        "spectator_count": websocket_manager.get_spectator_count(actual_game_id),
    }

    # Add trick info if in progress
    if trick:
        spectator_state["current_trick"] = {
            "number": trick.number,
            "cards": [
                {"player_id": pc.player_id, "card_id": pc.card_id.value}
                for pc in trick.picked_cards
            ],
            "picking_player_id": trick.picking_player_id,
        }

    # Send initial state
    init_message = ServerMessage(
        command=Command.INIT,
        game_id=actual_game_id,
        content={"game": spectator_state, "is_spectator": True},
    )
    await websocket.send_json(init_message.to_dict())

    # Notify players that a spectator joined
    spectator_joined_message = ServerMessage(
        command=Command.SPECTATOR_JOINED,
        game_id=actual_game_id,
        content={
            "spectator_id": spectator_id,
            "username": username,
            "spectator_count": websocket_manager.get_spectator_count(actual_game_id),
        },
    )
    await websocket_manager.broadcast_to_game(spectator_joined_message, actual_game_id)

    # Handle spectator connection (mostly just keeping it alive)
    await websocket_manager.handle_spectator_message(websocket, actual_game_id, spectator_id)


@router.get("/games/cards")
async def get_cards() -> CardListResponse:
    """Get all cards in the deck.

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


@router.get("/games/active")
async def get_active_games() -> dict[str, Any]:
    """Get list of active games that can be joined or spectated.

    Returns:
        List of active games with player counts and states.
        Games in PENDING state can be joined, others can only be spectated.

    """
    active_games = []

    for game_id, game in websocket_manager.games.items():
        # Only include games that are in progress or have players
        if game.state != GameState.ENDED and len(game.players) > 0:
            is_joinable = game.state == GameState.PENDING
            active_games.append(
                {
                    "game_id": game_id,
                    "slug": game.slug,
                    "state": game.state.value,
                    "joinable": is_joinable,
                    "player_count": len(game.players),
                    "max_players": 6,  # Skull King supports up to 6 players
                    "player_names": [p.username for p in game.players if not p.is_bot][:3],
                    "bot_count": sum(1 for p in game.players if p.is_bot),
                    "current_round": game.current_round_number,
                    "spectator_count": websocket_manager.get_spectator_count(game_id),
                }
            )

    # Sort joinable games first, then by player count descending
    active_games.sort(key=lambda g: (not g["joinable"], -g["player_count"]))

    return {"games": active_games, "count": len(active_games)}


@router.get("/games/{game_id}")
async def get_game(game_id: str) -> dict[str, Any]:
    """Get game state.

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


# ============================================
#  Game History & Replay Endpoints
# ============================================


@router.get("/games/history")
async def get_game_history_list(limit: Annotated[int, Query(le=50)] = 10) -> dict[str, Any]:
    """Get list of recently completed games.

    Args:
        limit: Maximum number of games to return (max 50)

    Returns:
        List of game summaries

    """
    histories = event_recorder.get_recent_histories(limit)
    return {"games": histories, "count": len(histories)}


@router.get("/games/{game_id}/replay")
async def get_game_replay(game_id: str) -> dict[str, Any]:
    """Get full replay data for a completed game.

    Args:
        game_id: ID of the completed game

    Returns:
        Complete game history with all events for replay

    """
    history = event_recorder.get_history(game_id)

    if not history:
        raise HTTPException(
            status_code=404,
            detail="Game history not found. Game may still be in progress or was not recorded.",
        )

    return history.to_dict()
