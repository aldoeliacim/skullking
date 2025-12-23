"""Pirate abilities system for Skull King.

Each of the 5 pirates has a unique ability that activates when winning a trick.

Pirates and their abilities:
1. Rosie de Laney - Choose who opens next trick
2. El bandido Bendt - Draw 2 cards, discard 2
3. Bribón de Roatán - Additional bet of 0/10/20 points
4. Juanita Jade - Look at undealt cards
5. Harry, el Gigante - Modify bid by +1/-1/0 at end of round
"""

from dataclasses import dataclass, field
from enum import Enum

from app.models.card import CardId


class PirateType(str, Enum):
    """Individual pirate identities."""

    ROSIE = "rosie"  # Rosie de Laney
    BENDT = "bendt"  # El bandido Bendt
    ROATAN = "roatan"  # Bribón de Roatán
    JADE = "jade"  # Juanita Jade
    HARRY = "harry"  # Harry, el Gigante


class AbilityType(str, Enum):
    """Types of pirate abilities."""

    CHOOSE_STARTER = "choose_starter"  # Rosie - choose who opens next trick
    DRAW_DISCARD = "draw_discard"  # Bendt - draw 2, discard 2
    EXTRA_BET = "extra_bet"  # Roatán - additional 0/10/20 bet
    VIEW_DECK = "view_deck"  # Jade - view undealt cards
    MODIFY_BID = "modify_bid"  # Harry - modify bid by ±1 at end


# Map pirate card IDs to their identity
PIRATE_IDENTITY: dict[CardId, PirateType] = {
    CardId.PIRATE1: PirateType.ROSIE,
    CardId.PIRATE2: PirateType.BENDT,
    CardId.PIRATE3: PirateType.ROATAN,
    CardId.PIRATE4: PirateType.JADE,
    CardId.PIRATE5: PirateType.HARRY,
}

# Map pirate identity to their ability
PIRATE_ABILITY: dict[PirateType, AbilityType] = {
    PirateType.ROSIE: AbilityType.CHOOSE_STARTER,
    PirateType.BENDT: AbilityType.DRAW_DISCARD,
    PirateType.ROATAN: AbilityType.EXTRA_BET,
    PirateType.JADE: AbilityType.VIEW_DECK,
    PirateType.HARRY: AbilityType.MODIFY_BID,
}


def get_pirate_type(card_id: CardId) -> PirateType | None:
    """Get the pirate type for a card ID."""
    return PIRATE_IDENTITY.get(card_id)


def get_ability_type(pirate: PirateType) -> AbilityType:
    """Get the ability type for a pirate."""
    return PIRATE_ABILITY[pirate]


def get_card_ability(card_id: CardId) -> AbilityType | None:
    """Get the ability type for a pirate card ID."""
    pirate = get_pirate_type(card_id)
    if pirate:
        return get_ability_type(pirate)
    return None


@dataclass
class PendingAbility:
    """Represents an ability waiting to be activated.

    Attributes:
        player_id: Player who triggered the ability
        pirate_type: Which pirate's ability
        ability_type: What kind of ability
        trick_number: Which trick triggered it (for timing)
        resolved: Whether the ability has been used
    """

    player_id: str
    pirate_type: PirateType
    ability_type: AbilityType
    trick_number: int
    resolved: bool = False
    # Additional data depending on ability type
    extra_bet: int | None = None  # For Roatán: 0, 10, or 20
    chosen_starter: str | None = None  # For Rosie: player_id to start next trick
    drawn_cards: list[CardId] = field(default_factory=list)  # For Bendt: cards drawn
    discarded_cards: list[CardId] = field(default_factory=list)  # For Bendt: cards discarded
    bid_modifier: int | None = None  # For Harry: -1, 0, or +1


@dataclass
class AbilityState:
    """Tracks pirate abilities for a round.

    Attributes:
        pending_abilities: Abilities waiting to be resolved
        harry_armed: Player IDs who won with Harry (activate at round end)
        roatan_bets: Extra bets from Roatán (player_id -> amount)
        rosie_next_starter: Override for who starts next trick
    """

    pending_abilities: list[PendingAbility] = field(default_factory=list)
    harry_armed: dict[str, bool] = field(default_factory=dict)
    roatan_bets: dict[str, int] = field(default_factory=dict)
    rosie_next_starter: str | None = None

    def trigger_ability(
        self, player_id: str, card_id: CardId, trick_number: int
    ) -> PendingAbility | None:
        """Trigger an ability when a player wins with a pirate.

        Args:
            player_id: Player who won the trick
            card_id: The pirate card that won
            trick_number: Current trick number

        Returns:
            PendingAbility if ability was triggered, None otherwise
        """
        pirate = get_pirate_type(card_id)
        if not pirate:
            return None

        ability_type = get_ability_type(pirate)

        # Harry is special - just arm it for later
        if ability_type == AbilityType.MODIFY_BID:
            self.harry_armed[player_id] = True
            return None  # No immediate action needed

        # Create pending ability for others
        pending = PendingAbility(
            player_id=player_id,
            pirate_type=pirate,
            ability_type=ability_type,
            trick_number=trick_number,
        )
        self.pending_abilities.append(pending)
        return pending

    def get_pending_ability(self, player_id: str) -> PendingAbility | None:
        """Get the oldest unresolved ability for a player."""
        for ability in self.pending_abilities:
            if ability.player_id == player_id and not ability.resolved:
                return ability
        return None

    def resolve_rosie(self, player_id: str, chosen_player_id: str) -> bool:
        """Resolve Rosie's ability - choose who starts next trick.

        Args:
            player_id: Player using the ability
            chosen_player_id: Player chosen to start next trick

        Returns:
            True if ability was resolved successfully
        """
        ability = self.get_pending_ability(player_id)
        if not ability or ability.ability_type != AbilityType.CHOOSE_STARTER:
            return False

        ability.chosen_starter = chosen_player_id
        ability.resolved = True
        self.rosie_next_starter = chosen_player_id
        return True

    def resolve_roatan(self, player_id: str, extra_bet: int) -> bool:
        """Resolve Roatán's ability - declare extra bet.

        Args:
            player_id: Player using the ability
            extra_bet: Amount to bet (0, 10, or 20)

        Returns:
            True if ability was resolved successfully
        """
        if extra_bet not in (0, 10, 20):
            return False

        ability = self.get_pending_ability(player_id)
        if not ability or ability.ability_type != AbilityType.EXTRA_BET:
            return False

        ability.extra_bet = extra_bet
        ability.resolved = True
        self.roatan_bets[player_id] = self.roatan_bets.get(player_id, 0) + extra_bet
        return True

    def resolve_bendt(
        self, player_id: str, drawn_cards: list[CardId], discarded_cards: list[CardId]
    ) -> bool:
        """Resolve Bendt's ability - draw 2, discard 2.

        Args:
            player_id: Player using the ability
            drawn_cards: Cards drawn from deck
            discarded_cards: Cards to discard from hand

        Returns:
            True if ability was resolved successfully
        """
        ability = self.get_pending_ability(player_id)
        if not ability or ability.ability_type != AbilityType.DRAW_DISCARD:
            return False

        if len(discarded_cards) != min(2, len(drawn_cards)):
            return False

        ability.drawn_cards = drawn_cards
        ability.discarded_cards = discarded_cards
        ability.resolved = True
        return True

    def resolve_jade(self, player_id: str) -> bool:
        """Resolve Jade's ability - just mark as viewed.

        The actual viewing is handled by the game handler (show undealt cards to player).

        Args:
            player_id: Player using the ability

        Returns:
            True if ability was resolved successfully
        """
        ability = self.get_pending_ability(player_id)
        if not ability or ability.ability_type != AbilityType.VIEW_DECK:
            return False

        ability.resolved = True
        return True

    def resolve_harry(self, player_id: str, modifier: int) -> bool:
        """Resolve Harry's ability - modify bid at end of round.

        Args:
            player_id: Player using the ability
            modifier: Bid modification (-1, 0, or +1)

        Returns:
            True if ability was resolved successfully
        """
        if modifier not in (-1, 0, 1):
            return False

        if player_id not in self.harry_armed or not self.harry_armed[player_id]:
            return False

        # Store the modifier - will be applied during scoring
        self.harry_armed[player_id] = False  # Mark as used

        # Create a pending ability to track the modifier
        pending = PendingAbility(
            player_id=player_id,
            pirate_type=PirateType.HARRY,
            ability_type=AbilityType.MODIFY_BID,
            trick_number=-1,  # End of round
            resolved=True,
            bid_modifier=modifier,
        )
        self.pending_abilities.append(pending)
        return True

    def get_harry_modifier(self, player_id: str) -> int:
        """Get the bid modifier for a player from Harry's ability.

        Returns:
            The bid modifier, or 0 if none
        """
        for ability in self.pending_abilities:
            if (
                ability.player_id == player_id
                and ability.ability_type == AbilityType.MODIFY_BID
                and ability.resolved
                and ability.bid_modifier is not None
            ):
                return ability.bid_modifier
        return 0

    def has_armed_harry(self, player_id: str) -> bool:
        """Check if player has Harry's ability armed."""
        return self.harry_armed.get(player_id, False)

    def get_roatan_bonus(self, player_id: str, bid_correct: bool) -> int:
        """Get the bonus/penalty from Roatán's extra bets.

        Args:
            player_id: Player to check
            bid_correct: Whether the player's bid was correct

        Returns:
            Positive bonus if bid correct, negative penalty if wrong
        """
        bet = self.roatan_bets.get(player_id, 0)
        return bet if bid_correct else -bet

    def clear_rosie_override(self) -> None:
        """Clear Rosie's next starter override after it's used."""
        self.rosie_next_starter = None

    def has_pending_abilities(self, player_id: str | None = None) -> bool:
        """Check if there are any unresolved abilities.

        Args:
            player_id: Optional player to filter by

        Returns:
            True if there are pending abilities
        """
        for ability in self.pending_abilities:
            if not ability.resolved and (player_id is None or ability.player_id == player_id):
                return True
        return False
