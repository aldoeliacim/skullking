"""Event recorder service for capturing game events during gameplay.

Used for replay and game history features.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.models.game_event import GameEvent, GameEventType, GameHistory

if TYPE_CHECKING:
    from app.models.game import Game


class EventRecorder:
    """Records game events for later replay."""

    def __init__(self) -> None:
        """Initialize the event recorder.

        Sets up in-memory storage for game events during gameplay and completed game histories.
        """
        # In-memory storage during game
        # Key: game_id, Value: list of events
        self._events: dict[str, list[GameEvent]] = {}
        self._game_start_times: dict[str, datetime] = {}
        # Completed game histories
        self._histories: dict[str, GameHistory] = {}

    def start_game(self, game: "Game") -> None:
        """Initialize event recording for a new game."""
        game_id = str(game.id)
        self._events[game_id] = []
        self._game_start_times[game_id] = datetime.now(UTC)

        self.record_event(
            game_id=game_id,
            event_type=GameEventType.GAME_STARTED,
            data={
                "players": [
                    {"id": p.id, "username": p.username, "index": p.index, "is_bot": p.is_bot}
                    for p in game.players
                ],
                "slug": game.slug,
            },
        )

    def record_event(  # noqa: PLR0913
        self,
        game_id: str,
        event_type: GameEventType,
        round_number: int = 0,
        trick_number: int | None = None,
        player_id: str | None = None,
        data: dict | None = None,
    ) -> None:
        """Record a single game event."""
        if game_id not in self._events:
            self._events[game_id] = []

        event = GameEvent(
            game_id=game_id,
            event_type=event_type,
            round_number=round_number,
            trick_number=trick_number,
            player_id=player_id,
            data=data or {},
        )
        self._events[game_id].append(event)

    def record_round_start(self, game: "Game") -> None:
        """Record round start with dealt cards."""
        current_round = game.get_current_round()
        if not current_round:
            return

        self.record_event(
            game_id=str(game.id),
            event_type=GameEventType.ROUND_STARTED,
            round_number=current_round.number,
            data={
                "starter_player_index": current_round.starter_player_index,
                # Store dealt cards for replay (hidden during live game)
                "dealt_cards": {
                    pid: [c.value for c in cards]
                    for pid, cards in current_round.dealt_cards.items()
                },
            },
        )

    def record_bid(self, game: "Game", player_id: str, bid: int) -> None:
        """Record a player's bid."""
        current_round = game.get_current_round()
        round_num = current_round.number if current_round else 0

        self.record_event(
            game_id=str(game.id),
            event_type=GameEventType.BID_PLACED,
            round_number=round_num,
            player_id=player_id,
            data={"bid": bid},
        )

    def record_card_played(
        self, game: "Game", player_id: str, card_id: int, tigress_choice: str | None = None
    ) -> None:
        """Record a card being played."""
        current_round = game.get_current_round()
        if not current_round:
            return

        trick = current_round.get_current_trick()
        trick_num = trick.number if trick else 0

        data = {"card_id": card_id}
        if tigress_choice:
            data["tigress_choice"] = tigress_choice

        self.record_event(
            game_id=str(game.id),
            event_type=GameEventType.CARD_PLAYED,
            round_number=current_round.number,
            trick_number=trick_num,
            player_id=player_id,
            data=data,
        )

    def record_trick_won(
        self, game: "Game", winner_id: str, winning_card_id: int, bonus_points: int = 0
    ) -> None:
        """Record trick winner."""
        current_round = game.get_current_round()
        if not current_round:
            return

        trick = current_round.get_current_trick()
        trick_num = trick.number if trick else 0

        self.record_event(
            game_id=str(game.id),
            event_type=GameEventType.TRICK_WON,
            round_number=current_round.number,
            trick_number=trick_num,
            player_id=winner_id,
            data={"winning_card_id": winning_card_id, "bonus_points": bonus_points},
        )

    def record_round_end(self, game: "Game", scores: list[dict]) -> None:
        """Record round completion with scores."""
        current_round = game.get_current_round()
        round_num = current_round.number if current_round else 0

        self.record_event(
            game_id=str(game.id),
            event_type=GameEventType.ROUND_ENDED,
            round_number=round_num,
            data={"scores": scores},
        )

    def record_scores(self, game: "Game", scores: list[dict]) -> None:
        """Record score announcement."""
        current_round = game.get_current_round()
        round_num = current_round.number if current_round else 0

        self.record_event(
            game_id=str(game.id),
            event_type=GameEventType.SCORES_ANNOUNCED,
            round_number=round_num,
            data={"scores": scores},
        )

    def end_game(self, game: "Game") -> GameHistory | None:
        """Finalize game recording and create history."""
        game_id = str(game.id)

        if game_id not in self._events:
            return None

        # Get final player scores
        players_final = [
            {
                "id": p.id,
                "username": p.username,
                "score": p.score,
                "index": p.index,
                "is_bot": p.is_bot,
            }
            for p in sorted(game.players, key=lambda x: x.score, reverse=True)
        ]

        winner = players_final[0] if players_final else {"id": "", "username": "Unknown"}

        self.record_event(
            game_id=game_id,
            event_type=GameEventType.GAME_ENDED,
            data={
                "final_scores": players_final,
                "winner_id": winner["id"],
                "winner_username": winner["username"],
            },
        )

        # Calculate duration
        start_time = self._game_start_times.get(game_id, datetime.now(UTC))
        end_time = datetime.now(UTC)
        duration = int((end_time - start_time).total_seconds())

        # Create history
        history = GameHistory(
            game_id=game_id,
            slug=game.slug,
            created_at=start_time,
            ended_at=end_time,
            duration_seconds=duration,
            players=players_final,
            winner_id=winner["id"],
            winner_username=winner["username"],
            total_rounds=len(game.rounds),
            events=self._events[game_id],
        )

        # Store history and clean up
        self._histories[game_id] = history
        del self._events[game_id]
        if game_id in self._game_start_times:
            del self._game_start_times[game_id]

        return history

    def get_history(self, game_id: str) -> GameHistory | None:
        """Get completed game history."""
        return self._histories.get(game_id)

    def get_all_histories(self) -> list[GameHistory]:
        """Get all completed game histories."""
        return list(self._histories.values())

    def get_recent_histories(self, limit: int = 10) -> list[dict]:
        """Get recent game summaries."""
        histories = sorted(self._histories.values(), key=lambda h: h.ended_at, reverse=True)[:limit]
        return [h.get_summary() for h in histories]


# Global event recorder instance
event_recorder = EventRecorder()
