"""Tests for API routes."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.api.websocket import websocket_manager
from app.models.enums import GameState
from app.models.game import Game
from app.models.player import Player


@pytest.fixture
def test_app():
    """Create a test FastAPI app without lifespan dependencies."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    # Clear any existing games before each test
    websocket_manager.games.clear()
    websocket_manager.active_connections.clear()
    with TestClient(test_app, raise_server_exceptions=False) as client:
        yield client


class TestCreateGame:
    """Tests for POST /games endpoint."""

    def test_create_game_success(self, client):
        """Creating a game should return game_id and slug."""
        response = client.post("/games", json={"lobby_id": "test-lobby"})
        assert response.status_code == 200

        data = response.json()
        assert "game_id" in data
        assert "slug" in data
        assert data["message"] == "Game created successfully"
        # Slug is first 4 hex chars of UUID uppercased
        assert len(data["slug"]) == 4
        assert data["slug"] == data["game_id"][:4].upper()

    def test_create_game_adds_to_manager(self, client):
        """Created game should be accessible via websocket manager."""
        response = client.post("/games", json={"lobby_id": "test-lobby"})
        data = response.json()

        game = websocket_manager.get_game(data["game_id"])
        assert game is not None
        assert game.slug == data["slug"]
        assert game.state == GameState.PENDING


class TestGetCards:
    """Tests for GET /games/cards endpoint."""

    def test_get_cards_success(self, client):
        """Cards endpoint should return all card definitions."""
        response = client.get("/games/cards")
        assert response.status_code == 200

        data = response.json()
        assert "cards" in data
        assert len(data["cards"]) > 0

        # Check card structure
        card = data["cards"][0]
        assert "id" in card
        assert "number" in card
        assert "type" in card
        assert "name" in card

    def test_get_cards_includes_special_cards(self, client):
        """Cards should include special cards like Skull King."""
        response = client.get("/games/cards")
        data = response.json()

        card_types = {card["type"] for card in data["cards"]}
        assert "king" in card_types
        assert "pirate" in card_types
        assert "mermaid" in card_types
        assert "escape" in card_types


class TestGetGame:
    """Tests for GET /games/{game_id} endpoint."""

    def test_get_game_success(self, client):
        """Getting an existing game should return game state."""
        # First create a game
        create_response = client.post("/games", json={"lobby_id": "test"})
        game_id = create_response.json()["game_id"]

        # Then get it
        response = client.get(f"/games/{game_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == game_id
        assert data["state"] == "PENDING"
        assert data["current_round"] == 0
        assert data["players"] == []

    def test_get_game_not_found(self, client):
        """Getting a non-existent game should return 404."""
        response = client.get("/games/non-existent-id")
        assert response.status_code == 404
        assert "Game not found" in response.json()["detail"]

    def test_get_game_with_players(self, client):
        """Game with players should return player info."""
        # Create game and add players directly
        game = Game(id="test-game-123", slug="test-game")
        for i in range(2):
            player = Player(
                id=f"player-{i}",
                username=f"Player {i}",
                game_id=game.id,
                avatar_id=i,
            )
            game.add_player(player)
        websocket_manager.add_game(game)

        response = client.get("/games/test-game-123")
        assert response.status_code == 200

        data = response.json()
        assert len(data["players"]) == 2
        assert data["players"][0]["id"] == "player-0"
        assert data["players"][0]["username"] == "Player 0"
        assert data["players"][1]["id"] == "player-1"


class TestGameStateTransitions:
    """Tests for game state in API responses."""

    def test_game_state_after_start(self, client):
        """Game state should reflect current phase."""
        # Create game with players
        game = Game(id="state-test-game", slug="state-test")
        for i in range(3):
            player = Player(
                id=f"player-{i}",
                username=f"Player {i}",
                game_id=game.id,
            )
            game.add_player(player)

        # Start the game
        game.state = GameState.BIDDING
        game.start_new_round()
        websocket_manager.add_game(game)

        response = client.get("/games/state-test-game")
        data = response.json()

        assert data["state"] == "BIDDING"
        assert data["current_round"] == 1

    def test_game_state_during_picking(self, client):
        """Game in picking phase should show correct state."""
        game = Game(id="picking-test", slug="picking-test")
        for i in range(2):
            game.add_player(Player(id=f"p{i}", username=f"P{i}", game_id=game.id))

        game.state = GameState.PICKING
        game.current_round_number = 3
        websocket_manager.add_game(game)

        response = client.get("/games/picking-test")
        data = response.json()

        assert data["state"] == "PICKING"
        assert data["current_round"] == 3
