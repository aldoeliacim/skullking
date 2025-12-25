"""Game serialization for MongoDB persistence.

Handles conversion between Game objects and MongoDB documents.
"""

from datetime import UTC, datetime
from typing import Any

from app.models.card import CardId
from app.models.enums import GameState
from app.models.game import Game
from app.models.pirate_ability import (
    AbilityState,
    AbilityType,
    PendingAbility,
    PirateType,
)
from app.models.player import Player
from app.models.round import Round
from app.models.trick import PickedCard, TigressChoice, Trick


def serialize_player(player: Player) -> dict[str, Any]:
    """Serialize a Player to a dictionary."""
    return {
        "id": player.id,
        "username": player.username,
        "game_id": player.game_id,
        "avatar_id": player.avatar_id,
        "score": player.score,
        "index": player.index,
        "is_connected": player.is_connected,
        "is_bot": player.is_bot,
        "hand": [int(card_id) for card_id in player.hand],
        "bid": player.bid,
        "tricks_won": player.tricks_won,
    }


def deserialize_player(data: dict[str, Any]) -> Player:
    """Deserialize a Player from a dictionary."""
    return Player(
        id=data["id"],
        username=data["username"],
        game_id=data.get("game_id", ""),
        avatar_id=data.get("avatar_id", 0),
        score=data.get("score", 0),
        index=data.get("index", 0),
        is_connected=data.get("is_connected", True),
        is_bot=data.get("is_bot", False),
        hand=[CardId(card_id) for card_id in data.get("hand", [])],
        bid=data.get("bid"),
        tricks_won=data.get("tricks_won", 0),
    )


def serialize_picked_card(picked: PickedCard) -> dict[str, Any]:
    """Serialize a PickedCard to a dictionary."""
    return {
        "player_id": picked.player_id,
        "card_id": int(picked.card_id),
        "tigress_choice": picked.tigress_choice.value if picked.tigress_choice else None,
    }


def deserialize_picked_card(data: dict[str, Any]) -> PickedCard:
    """Deserialize a PickedCard from a dictionary."""
    tigress_choice = None
    if data.get("tigress_choice"):
        tigress_choice = TigressChoice(data["tigress_choice"])
    return PickedCard(
        player_id=data["player_id"],
        card_id=CardId(data["card_id"]),
        tigress_choice=tigress_choice,
    )


def serialize_trick(trick: Trick) -> dict[str, Any]:
    """Serialize a Trick to a dictionary."""
    return {
        "number": trick.number,
        "starter_player_index": trick.starter_player_index,
        "picking_player_id": trick.picking_player_id,
        "picked_cards": [serialize_picked_card(pc) for pc in trick.picked_cards],
        "winner_player_id": trick.winner_player_id,
        "winner_card_id": int(trick.winner_card_id) if trick.winner_card_id else None,
    }


def deserialize_trick(data: dict[str, Any]) -> Trick:
    """Deserialize a Trick from a dictionary."""
    return Trick(
        number=data["number"],
        starter_player_index=data["starter_player_index"],
        picking_player_id=data.get("picking_player_id", ""),
        picked_cards=[deserialize_picked_card(pc) for pc in data.get("picked_cards", [])],
        winner_player_id=data.get("winner_player_id"),
        winner_card_id=CardId(data["winner_card_id"]) if data.get("winner_card_id") else None,
    )


def serialize_pending_ability(ability: PendingAbility) -> dict[str, Any]:
    """Serialize a PendingAbility to a dictionary."""
    return {
        "player_id": ability.player_id,
        "pirate_type": ability.pirate_type.value,
        "ability_type": ability.ability_type.value,
        "trick_number": ability.trick_number,
        "resolved": ability.resolved,
        "extra_bet": ability.extra_bet,
        "chosen_starter": ability.chosen_starter,
        "drawn_cards": [int(c) for c in ability.drawn_cards],
        "discarded_cards": [int(c) for c in ability.discarded_cards],
        "bid_modifier": ability.bid_modifier,
    }


def deserialize_pending_ability(data: dict[str, Any]) -> PendingAbility:
    """Deserialize a PendingAbility from a dictionary."""
    return PendingAbility(
        player_id=data["player_id"],
        pirate_type=PirateType(data["pirate_type"]),
        ability_type=AbilityType(data["ability_type"]),
        trick_number=data["trick_number"],
        resolved=data.get("resolved", False),
        extra_bet=data.get("extra_bet"),
        chosen_starter=data.get("chosen_starter"),
        drawn_cards=[CardId(c) for c in data.get("drawn_cards", [])],
        discarded_cards=[CardId(c) for c in data.get("discarded_cards", [])],
        bid_modifier=data.get("bid_modifier"),
    )


def serialize_ability_state(state: AbilityState) -> dict[str, Any]:
    """Serialize AbilityState to a dictionary."""
    return {
        "pending_abilities": [serialize_pending_ability(a) for a in state.pending_abilities],
        "harry_armed": state.harry_armed,
        "roatan_bets": state.roatan_bets,
        "rosie_next_starter": state.rosie_next_starter,
    }


def deserialize_ability_state(data: dict[str, Any]) -> AbilityState:
    """Deserialize AbilityState from a dictionary."""
    return AbilityState(
        pending_abilities=[
            deserialize_pending_ability(a) for a in data.get("pending_abilities", [])
        ],
        harry_armed=data.get("harry_armed", {}),
        roatan_bets=data.get("roatan_bets", {}),
        rosie_next_starter=data.get("rosie_next_starter"),
    )


def serialize_round(round_obj: Round) -> dict[str, Any]:
    """Serialize a Round to a dictionary."""
    return {
        "number": round_obj.number,
        "starter_player_index": round_obj.starter_player_index,
        "dealt_cards": {
            player_id: [int(c) for c in cards]
            for player_id, cards in round_obj.dealt_cards.items()
        },
        "bids": round_obj.bids,
        "tricks": [serialize_trick(t) for t in round_obj.tricks],
        "scores": round_obj.scores,
        "ability_state": serialize_ability_state(round_obj.ability_state),
    }


def deserialize_round(data: dict[str, Any]) -> Round:
    """Deserialize a Round from a dictionary."""
    return Round(
        number=data["number"],
        starter_player_index=data["starter_player_index"],
        dealt_cards={
            player_id: [CardId(c) for c in cards]
            for player_id, cards in data.get("dealt_cards", {}).items()
        },
        bids=data.get("bids", {}),
        tricks=[deserialize_trick(t) for t in data.get("tricks", [])],
        scores=data.get("scores", {}),
        ability_state=deserialize_ability_state(data.get("ability_state", {})),
    )


def serialize_game(game: Game) -> dict[str, Any]:
    """Serialize a complete Game to a MongoDB document.

    Args:
        game: Game instance to serialize

    Returns:
        Dictionary suitable for MongoDB storage
    """
    return {
        "_id": game.id,
        "slug": game.slug,
        "state": game.state.value,
        "players": [serialize_player(p) for p in game.players],
        "rounds": [serialize_round(r) for r in game.rounds],
        "current_round_number": game.current_round_number,
        "created_at": game.created_at or datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
    }


def deserialize_game(data: dict[str, Any]) -> Game:
    """Deserialize a Game from a MongoDB document.

    Args:
        data: MongoDB document

    Returns:
        Game instance with full state restored
    """
    game = Game(
        id=data["_id"],
        slug=data["slug"],
        state=GameState(data["state"]),
        current_round_number=data.get("current_round_number", 0),
        created_at=data.get("created_at"),
    )

    # Restore players
    game.players = [deserialize_player(p) for p in data.get("players", [])]

    # Restore rounds
    game.rounds = [deserialize_round(r) for r in data.get("rounds", [])]

    return game
