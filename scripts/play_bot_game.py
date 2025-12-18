#!/usr/bin/env python3
"""
CLI script to watch bots play Skull King.

This script creates a game with multiple bot players and simulates
a complete game, displaying the results.
"""

import asyncio
import random
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.bots import RandomBot, RuleBasedBot
from app.models.card import get_card
from app.models.enums import GameState
from app.models.game import Game
from app.models.player import Player
from app.models.trick import Trick


class BotGameSimulator:
    """Simulates a game between bot players."""

    def __init__(self, num_players: int = 4, bot_types: list[str] = None):
        """
        Initialize simulator.

        Args:
            num_players: Number of players (2-7)
            bot_types: List of bot types for each player ("random" or "rule_based")
        """
        if not (2 <= num_players <= 7):
            raise ValueError("Must have 2-7 players")

        self.num_players = num_players
        self.bot_types = bot_types or ["rule_based"] * num_players
        self.game = None
        self.bots = []

    def setup_game(self) -> None:
        """Set up the game with bot players."""
        self.game = Game(id="bot_game_001", slug="bot-game-001")

        print(f"\n{'='*60}")
        print(f"Setting up Skull King game with {self.num_players} bots")
        print(f"{'='*60}\n")

        # Create bot players
        for i in range(self.num_players):
            player_id = f"bot_{i}"
            bot_type = self.bot_types[i] if i < len(self.bot_types) else "rule_based"

            player = Player(
                id=player_id,
                username=f"Bot{i+1}",
                game_id=self.game.id,
                index=i,
                is_bot=True,
            )
            self.game.add_player(player)

            # Create bot controller
            if bot_type == "random":
                bot = RandomBot(player_id)
            else:
                bot = RuleBasedBot(player_id)

            self.bots.append((player_id, bot))
            print(f"  Player {i+1}: {player.username} ({bot})")

        print()

    def play_round(self, round_number: int) -> None:
        """Play a single round."""
        print(f"\n{'='*60}")
        print(f"ROUND {round_number}")
        print(f"{'='*60}\n")

        # Start round
        self.game.start_new_round()
        self.game.deal_cards()

        current_round = self.game.get_current_round()
        if not current_round:
            return

        # Bidding phase
        print("BIDDING PHASE")
        print("-" * 40)

        self.game.state = GameState.BIDDING

        for player in self.game.players:
            # Find bot for this player
            bot = next((b for pid, b in self.bots if pid == player.id), None)
            if not bot:
                continue

            bid = bot.make_bid(self.game, round_number, player.hand)
            player.bid = bid
            current_round.add_bid(player.id, bid)

            print(f"{player.username} bids: {bid}")
            time.sleep(0.3)

        print()

        # Playing phase
        self.game.state = GameState.PICKING

        for trick_num in range(1, round_number + 1):
            self.play_trick(trick_num)

        # Calculate scores
        current_round.calculate_scores()

        print("\nROUND RESULTS")
        print("-" * 40)

        for player in self.game.players:
            score_delta = current_round.scores.get(player.id, 0)
            tricks_won = current_round.get_tricks_won(player.id)
            player.update_score(score_delta)

            result = "✓" if player.bid == tricks_won else "✗"
            print(
                f"{player.username}: Bid {player.bid}, Won {tricks_won} {result} | "
                f"Score: {score_delta:+d} (Total: {player.score})"
            )

        time.sleep(1)

    def play_trick(self, trick_number: int) -> None:
        """Play a single trick."""
        print(f"\n  Trick {trick_number}:")
        print(f"  {'-' * 36}")

        current_round = self.game.get_current_round()
        if not current_round:
            return

        # Determine starter
        if trick_number == 1:
            starter_index = current_round.starter_player_index
        else:
            # Winner of last trick starts
            last_trick = current_round.tricks[-1]
            if last_trick.winner_player_id:
                winner = self.game.get_player(last_trick.winner_player_id)
                starter_index = winner.index if winner else 0
            else:
                starter_index = 0

        # Create trick
        trick = Trick(
            number=trick_number,
            starter_player_index=starter_index,
        )
        current_round.tricks.append(trick)

        # Each player plays a card
        for i in range(self.num_players):
            player_index = (starter_index + i) % self.num_players
            player = self.game.players[player_index]

            # Find bot
            bot = next((b for pid, b in self.bots if pid == player.id), None)
            if not bot:
                continue

            # Bot picks card
            cards_in_trick = trick.get_all_card_ids()
            card_id = bot.pick_card(self.game, player.hand, cards_in_trick)

            # Play card
            player.remove_card(card_id)
            trick.add_card(player.id, card_id)

            card = get_card(card_id)
            print(f"    {player.username}: {card}")

            time.sleep(0.4)

        # Determine winner
        winner_card_id, winner_player_id = trick.determine_winner()

        if winner_player_id:
            winner = self.game.get_player(winner_player_id)
            winner_card = get_card(winner_card_id) if winner_card_id else None
            winner.tricks_won += 1
            print(f"  → Winner: {winner.username} with {winner_card}")
        else:
            print(f"  → Kraken! No one wins this trick")

        time.sleep(0.5)

    def play_game(self) -> None:
        """Play a complete game."""
        start_time = time.time()

        self.setup_game()
        self.game.state = GameState.DEALING

        # Play 10 rounds
        for round_num in range(1, 11):
            self.play_round(round_num)

        # Game over
        self.game.state = GameState.ENDED

        elapsed_time = time.time() - start_time

        print(f"\n{'='*60}")
        print("GAME OVER")
        print(f"{'='*60}\n")

        # Final standings
        leaderboard = self.game.get_leaderboard()

        print("FINAL STANDINGS:")
        print("-" * 40)

        for rank, entry in enumerate(leaderboard, 1):
            player = self.game.get_player(entry["player_id"])
            if player:
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
                print(f"{medal} {rank}. {player.username}: {player.score} points")

        winner = self.game.get_winner()
        if winner:
            print(f"\n🎉 Winner: {winner.username} with {winner.score} points! 🎉")

        print(f"\nGame duration: {elapsed_time:.1f} seconds")
        print(f"{'='*60}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Watch bots play Skull King")
    parser.add_argument(
        "--players",
        type=int,
        default=4,
        help="Number of players (2-7)",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        help="Number of random bots (rest will be rule-based)",
    )

    args = parser.parse_args()

    if not (2 <= args.players <= 7):
        print("Error: Must have 2-7 players")
        sys.exit(1)

    # Set up bot types
    bot_types = []
    for i in range(args.players):
        if i < args.random:
            bot_types.append("random")
        else:
            bot_types.append("rule_based")

    # Run simulation
    simulator = BotGameSimulator(num_players=args.players, bot_types=bot_types)
    simulator.play_game()


if __name__ == "__main__":
    main()
