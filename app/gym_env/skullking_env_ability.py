"""Ability-aware Gymnasium environment for Skull King.

Extends WorkerEnv to include pirate ability decisions as additional decision phases.
The agent learns when to use abilities strategically, not just card play.

Decision Phases:
- PLAYING: Select card to play (existing WorkerEnv behavior)
- ABILITY_ROSIE: Choose who starts next trick (0-5 player indices)
- ABILITY_BENDT_1: Choose first card to discard (0-10 card indices)
- ABILITY_BENDT_2: Choose second card to discard (0-10 card indices)
- ABILITY_ROATAN: Choose extra bet amount (0=0pts, 1=10pts, 2=20pts)
- ABILITY_HARRY: Choose bid modifier (0=-1, 1=0, 2=+1)

Observation space extends WorkerEnv with:
- Phase indicator (6 dims): one-hot for current decision phase
- Ability context (34 dims): phase-specific information

Action space: Discrete(11) with phase-specific masking
"""

import logging
from enum import IntEnum
from typing import Any

import numpy as np
from gymnasium import spaces
from sb3_contrib.common.wrappers import ActionMasker

from app.gym_env.skullking_env_hierarchical import WorkerEnv
from app.models.card import CardId, get_card
from app.models.enums import MAX_ROUNDS
from app.models.pirate_ability import (
    AbilityType,
    PendingAbility,
    get_pirate_type,
)
from app.models.trick import TigressChoice

logger = logging.getLogger(__name__)


class DecisionPhase(IntEnum):
    """Decision phases for ability-aware environment."""

    PLAYING = 0
    ABILITY_ROSIE = 1
    ABILITY_BENDT_1 = 2
    ABILITY_BENDT_2 = 3
    ABILITY_ROATAN = 4
    ABILITY_HARRY = 5


# Map ability types to decision phases
ABILITY_TO_PHASE = {
    AbilityType.CHOOSE_STARTER: DecisionPhase.ABILITY_ROSIE,
    AbilityType.DRAW_DISCARD: DecisionPhase.ABILITY_BENDT_1,
    AbilityType.EXTRA_BET: DecisionPhase.ABILITY_ROATAN,
    AbilityType.MODIFY_BID: DecisionPhase.ABILITY_HARRY,
}


class AbilityAwareEnv(WorkerEnv):
    """Environment that includes pirate ability decisions.

    Extends WorkerEnv to handle ability phases as additional decision points.
    When a pirate wins a trick and triggers an ability, the env transitions
    to the appropriate ability phase for the agent to make a decision.

    Observation breakdown (243 dims = 203 base + 40 ability):
    - Base WorkerEnv observations (203 dims)
    - Phase one-hot (6 dims)
    - Ability context (34 dims):
      - Rosie: player positions/strengths (6)
      - Bendt: drawn cards (18), discard state (2)
      - Roatán: bid confidence (2)
      - Harry: bid/tricks state (4)
      - Reserved (2)
    """

    # Additional observation dimensions
    # Note: Don't use PHASE_DIM here - it would override parent's PHASE_DIM = 3
    DECISION_PHASE_DIM = 6  # One-hot for decision phase (our 6 phases)
    ABILITY_CONTEXT_DIM = 34  # Phase-specific context

    def __init__(
        self,
        num_opponents: int = 3,
        opponent_bot_type: str = "rule_based",
        opponent_difficulty: str = "medium",
        fixed_goal: int | None = None,
        use_weighted_sampling: bool = True,
        allowed_phases: tuple[int, ...] | None = None,
        enable_abilities: bool = True,
    ) -> None:
        """Initialize ability-aware environment.

        Args:
            num_opponents: Number of bot opponents
            opponent_bot_type: Type of opponent bot
            opponent_difficulty: Difficulty level
            fixed_goal: If set, use this bid goal
            use_weighted_sampling: If True, sample rounds proportional to weights
            allowed_phases: Tuple of allowed phase indices for curriculum
            enable_abilities: If False, auto-resolve abilities (for comparison)
        """
        super().__init__(
            num_opponents=num_opponents,
            opponent_bot_type=opponent_bot_type,
            opponent_difficulty=opponent_difficulty,
            fixed_goal=fixed_goal,
            use_weighted_sampling=use_weighted_sampling,
            allowed_phases=allowed_phases,
        )

        self.enable_abilities = enable_abilities

        # Extended observation space
        self.OBS_DIM_EXTENDED = self.OBS_DIM + self.DECISION_PHASE_DIM + self.ABILITY_CONTEXT_DIM
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.OBS_DIM_EXTENDED,), dtype=np.float32
        )

        # Decision phase state
        self.decision_phase = DecisionPhase.PLAYING
        self.pending_ability: PendingAbility | None = None

        # Bendt-specific state
        self.bendt_drawn_cards: list[CardId] = []
        self.bendt_first_discard: CardId | None = None

        # Harry state (resolved at round end)
        self.harry_pending = False

        # Metrics
        self.ability_decisions: dict[str, list[int]] = {
            "rosie": [],
            "bendt": [],
            "roatan": [],
            "harry": [],
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to start of card-play phase."""
        obs, info = super().reset(seed=seed, options=options)

        # Reset ability state
        self.decision_phase = DecisionPhase.PLAYING
        self.pending_ability = None
        self.bendt_drawn_cards = []
        self.bendt_first_discard = None
        self.harry_pending = False

        # Reset metrics
        for key in self.ability_decisions:
            self.ability_decisions[key] = []

        # Extend observation with ability context
        extended_obs = self._extend_observation(obs)
        return extended_obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute action based on current decision phase.

        Routes to appropriate handler based on phase:
        - PLAYING: Play a card (delegates to parent)
        - ABILITY_*: Resolve the specific ability

        Args:
            action: Action index (meaning depends on phase)

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Dispatch based on decision phase
        phase_handlers = {
            DecisionPhase.PLAYING: self._step_playing,
            DecisionPhase.ABILITY_ROSIE: self._step_rosie,
            DecisionPhase.ABILITY_BENDT_1: self._step_bendt_1,
            DecisionPhase.ABILITY_BENDT_2: self._step_bendt_2,
            DecisionPhase.ABILITY_ROATAN: self._step_roatan,
            DecisionPhase.ABILITY_HARRY: self._step_harry,
        }

        handler = phase_handlers.get(self.decision_phase)
        if handler:
            return handler(action)

        # Should never reach here
        logger.error("Unknown decision phase: %s", self.decision_phase)
        return self._get_extended_obs(), 0.0, True, False, {}

    def _step_playing(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Handle card play action - extends parent with ability triggering."""
        current_round = self.game.get_current_round()
        if not current_round:
            return self._get_extended_obs(), 0.0, True, False, {}

        agent = self.game.get_player(self.agent_player_id)
        if not agent or not agent.hand:
            return self._get_extended_obs(), 0.0, True, False, {}

        # Get or create current trick
        trick = current_round.get_current_trick()
        if not trick or trick.is_complete(self.num_players):
            trick = self._start_new_trick(current_round)

        if not trick:
            return self._get_extended_obs(), 0.0, True, False, {}

        # Play cards up to agent's turn
        self._play_until_agent_turn(current_round, trick)

        # Play agent's card
        valid_cards = trick.get_valid_cards(agent.hand, trick.get_all_card_ids())

        if 0 <= action < len(agent.hand):
            card_id = agent.hand[action]
            if card_id not in valid_cards:
                card_id = valid_cards[0] if valid_cards else agent.hand[0]
        else:
            card_id = valid_cards[0] if valid_cards else agent.hand[0]

        # Handle Tigress choice
        tigress_choice = None
        card = get_card(card_id)
        if card.is_tigress():
            need_wins = self.goal_bid - self.tricks_won
            tigress_choice = TigressChoice.PIRATE if need_wins > 0 else TigressChoice.ESCAPE

        trick.add_card(self.agent_player_id, card_id, tigress_choice)
        agent.hand.remove(card_id)

        # Complete trick with remaining players
        self._complete_trick(current_round, trick)

        # Determine winner
        trick.determine_winner()
        won_trick = trick.winner_player_id == self.agent_player_id
        if won_trick:
            self.tricks_won += 1

        # Calculate base reward
        reward = self._calculate_worker_reward(won_trick, trick)

        # Check for ability trigger if agent won with a pirate
        if won_trick and self.enable_abilities:
            played_card = card_id
            pirate_type = get_pirate_type(played_card)
            if pirate_type:
                # Trigger ability
                ability = current_round.ability_state.trigger_ability(
                    self.agent_player_id, played_card, len(current_round.tricks)
                )

                if ability:
                    # Set up ability phase
                    self.pending_ability = ability
                    self.decision_phase = ABILITY_TO_PHASE.get(
                        ability.ability_type, DecisionPhase.PLAYING
                    )

                    # Special handling for Bendt - draw cards now
                    if ability.ability_type == AbilityType.DRAW_DISCARD:
                        self.bendt_drawn_cards = self._draw_cards_for_bendt()
                        agent.hand.extend(self.bendt_drawn_cards)

                    # Return with ability phase active - don't end episode yet
                    obs = self._get_extended_obs()
                    info = {
                        "won_trick": won_trick,
                        "tricks_won": self.tricks_won,
                        "goal_bid": self.goal_bid,
                        "ability_triggered": ability.ability_type.value,
                        "decision_phase": self.decision_phase.name,
                    }
                    return obs, reward, False, False, info

                # Harry is special - armed for end of round
                if pirate_type.value == "harry":
                    self.harry_pending = True

        # Check if round is complete
        round_complete = (
            len(current_round.tricks) >= self.current_round_num
            and trick.is_complete(self.num_players)
        )

        # Handle Harry at round end
        if round_complete and self.harry_pending and self.enable_abilities:
            self.decision_phase = DecisionPhase.ABILITY_HARRY
            obs = self._get_extended_obs()
            info = {
                "won_trick": won_trick,
                "tricks_won": self.tricks_won,
                "goal_bid": self.goal_bid,
                "ability_triggered": "harry_end_of_round",
                "decision_phase": self.decision_phase.name,
            }
            return obs, reward, False, False, info

        if round_complete:
            current_round.calculate_scores()

        info = {
            "won_trick": won_trick,
            "tricks_won": self.tricks_won,
            "goal_bid": self.goal_bid,
            "tricks_remaining": self.current_round_num - len(current_round.tricks),
            "goal_achieved": self.tricks_won == self.goal_bid if round_complete else None,
            "ability_decisions": self.ability_decisions.copy(),
        }

        return self._get_extended_obs(), reward, round_complete, False, info

    def _step_rosie(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Handle Rosie's choose-starter ability."""
        current_round = self.game.get_current_round()
        if not current_round or not self.pending_ability:
            self.decision_phase = DecisionPhase.PLAYING
            return self._get_extended_obs(), 0.0, False, False, {}

        # Map action to player ID
        num_players = len(self.game.players)
        player_idx = min(action, num_players - 1)
        chosen_player = self.game.players[player_idx]

        # Resolve ability
        current_round.ability_state.resolve_rosie(
            self.agent_player_id, chosen_player.id
        )

        # Track decision
        self.ability_decisions["rosie"].append(action)

        # Calculate reward - choosing self is often good strategically
        reward = 0.1 if chosen_player.id == self.agent_player_id else -0.05

        # Return to playing phase
        self.decision_phase = DecisionPhase.PLAYING
        self.pending_ability = None

        return self._continue_after_ability(reward)

    def _step_bendt_1(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Handle Bendt's first discard selection."""
        agent = self.game.get_player(self.agent_player_id)
        if not agent or not agent.hand:
            self.decision_phase = DecisionPhase.PLAYING
            return self._get_extended_obs(), 0.0, False, False, {}

        # Map action to card
        card_to_discard = (
            agent.hand[action] if 0 <= action < len(agent.hand) else agent.hand[0]
        )

        # Store first discard choice
        self.bendt_first_discard = card_to_discard

        # Track decision
        self.ability_decisions["bendt"].append(action)

        # Move to second discard phase
        self.decision_phase = DecisionPhase.ABILITY_BENDT_2

        # Small reward based on discard quality
        card = get_card(card_to_discard)
        reward = self._evaluate_bendt_discard(card)

        return self._get_extended_obs(), reward, False, False, {
            "decision_phase": self.decision_phase.name,
            "first_discard": card_to_discard.value,
        }

    def _step_bendt_2(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Handle Bendt's second discard selection."""
        current_round = self.game.get_current_round()
        agent = self.game.get_player(self.agent_player_id)
        if not current_round or not agent or not agent.hand:
            self.decision_phase = DecisionPhase.PLAYING
            return self._get_extended_obs(), 0.0, False, False, {}

        # Map action to card (excluding first discard)
        valid_indices = [i for i, c in enumerate(agent.hand) if c != self.bendt_first_discard]
        if action in valid_indices:
            card_to_discard = agent.hand[action]
        elif valid_indices:
            card_to_discard = agent.hand[valid_indices[0]]
        else:
            card_to_discard = agent.hand[0]

        # Track decision
        self.ability_decisions["bendt"].append(action)

        # Actually discard both cards
        discards = [self.bendt_first_discard, card_to_discard]
        for card_id in discards:
            if card_id and card_id in agent.hand:
                agent.hand.remove(card_id)

        # Resolve ability
        if self.pending_ability:
            current_round.ability_state.resolve_bendt(
                self.agent_player_id,
                self.bendt_drawn_cards,
                [c for c in discards if c],
            )

        # Calculate reward for second discard
        card = get_card(card_to_discard)
        reward = self._evaluate_bendt_discard(card)

        # Clean up
        self.decision_phase = DecisionPhase.PLAYING
        self.pending_ability = None
        self.bendt_drawn_cards = []
        self.bendt_first_discard = None

        return self._continue_after_ability(reward)

    def _step_roatan(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Handle Roatán's extra bet selection."""
        current_round = self.game.get_current_round()
        if not current_round or not self.pending_ability:
            self.decision_phase = DecisionPhase.PLAYING
            return self._get_extended_obs(), 0.0, False, False, {}

        # Map action to bet amount
        bet_amounts = [0, 10, 20]
        bet = bet_amounts[min(action, 2)]

        # Resolve ability
        current_round.ability_state.resolve_roatan(self.agent_player_id, bet)

        # Track decision
        self.ability_decisions["roatan"].append(action)

        # Reward will be applied at round end based on bid accuracy
        # For now, give small signal based on risk assessment
        tricks_needed = self.goal_bid - self.tricks_won
        tricks_remaining = self.current_round_num - len(current_round.tricks)

        if tricks_needed <= 0:
            # Already at goal - higher bet is good
            reward = bet * 0.005
        elif tricks_needed > tricks_remaining:
            # Can't make goal - lower bet is safer
            reward = -bet * 0.005
        else:
            # Uncertain - neutral
            reward = 0.0

        # Return to playing phase
        self.decision_phase = DecisionPhase.PLAYING
        self.pending_ability = None

        return self._continue_after_ability(reward)

    def _step_harry(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Handle Harry's bid modification at round end."""
        current_round = self.game.get_current_round()
        if not current_round:
            return self._get_extended_obs(), 0.0, True, False, {}

        # Map action to modifier
        modifiers = [-1, 0, 1]
        modifier = modifiers[min(action, 2)]

        # Apply modifier to bid
        agent = self.game.get_player(self.agent_player_id)
        if agent:
            original_bid = agent.bid or 0

            # Resolve ability (handles bid modification internally)
            current_round.ability_state.resolve_harry(self.agent_player_id, modifier)

            # Track decision
            self.ability_decisions["harry"].append(action)

            # Calculate reward based on whether modifier helps.
            # Positive bid_diff means won too many tricks (need +1 bid).
            # Negative bid_diff means won too few tricks (need -1 bid).
            bid_diff = self.tricks_won - original_bid

            if modifier == 0:
                # No change - good if already matching
                reward = 0.1 if bid_diff == 0 else 0.0
            elif (bid_diff == 1 and modifier == 1) or (bid_diff == -1 and modifier == -1):
                # Modifier saves the bid! (+1 trick needs +1 bid, -1 trick needs -1 bid)
                reward = 0.3
            elif (bid_diff == 1 and modifier == -1) or (bid_diff == -1 and modifier == 1):
                # Modifier goes wrong direction - makes it worse
                reward = -0.2
            else:
                reward = 0.0

        else:
            reward = 0.0

        # Calculate final scores
        current_round.calculate_scores()

        # Episode complete
        self.decision_phase = DecisionPhase.PLAYING
        self.harry_pending = False

        info = {
            "tricks_won": self.tricks_won,
            "goal_bid": self.goal_bid,
            "goal_achieved": self.tricks_won == (agent.bid if agent else 0),
            "harry_modifier": modifier,
            "ability_decisions": self.ability_decisions.copy(),
        }

        return self._get_extended_obs(), reward, True, False, info

    def _continue_after_ability(
        self, ability_reward: float = 0.0
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Continue game after ability resolution - start next trick or end round.

        Args:
            ability_reward: Reward earned from the ability decision
        """
        current_round = self.game.get_current_round()
        if not current_round:
            return self._get_extended_obs(), ability_reward, True, False, {}

        # Check if round is complete
        round_complete = len(current_round.tricks) >= self.current_round_num

        # Handle Harry at round end
        if round_complete and self.harry_pending and self.enable_abilities:
            self.decision_phase = DecisionPhase.ABILITY_HARRY
            return self._get_extended_obs(), ability_reward, False, False, {
                "decision_phase": self.decision_phase.name,
            }

        if round_complete:
            current_round.calculate_scores()

        info = {
            "tricks_won": self.tricks_won,
            "goal_bid": self.goal_bid,
            "tricks_remaining": self.current_round_num - len(current_round.tricks),
            "goal_achieved": self.tricks_won == self.goal_bid if round_complete else None,
        }

        return self._get_extended_obs(), ability_reward, round_complete, False, info

    def _draw_cards_for_bendt(self) -> list[CardId]:
        """Draw cards for Bendt's ability from remaining deck."""
        if not self.game:
            return []

        # Get cards not in any player's hand and not played
        all_cards = set(CardId)
        used_cards: set[CardId] = set()

        for player in self.game.players:
            used_cards.update(player.hand)

        current_round = self.game.get_current_round()
        if current_round:
            for trick in current_round.tricks:
                for pc in trick.picked_cards:
                    used_cards.add(pc.card_id)

        available = list(all_cards - used_cards)

        # Draw up to 2 cards
        num_draw = min(2, len(available))
        if num_draw == 0:
            return []

        drawn_indices = self.np_random.choice(len(available), size=num_draw, replace=False)
        return [available[i] for i in drawn_indices]

    def _evaluate_bendt_discard(self, card) -> float:
        """Evaluate quality of discarding a card for Bendt's ability.

        Good discards (positive reward):
        - Escape cards when bid > 0 (don't need to lose tricks)
        - Low numbered cards (weak)

        Bad discards (negative reward):
        - High cards when bid > 0 (might need them to win)
        - Pirates/specials (usually valuable)
        """
        tricks_needed = self.goal_bid - self.tricks_won

        if card.is_escape():
            # Escapes are good to discard if we need wins
            return 0.1 if tricks_needed > 0 else -0.05

        if card.is_pirate() or card.is_king() or card.is_mermaid():
            # Specials are usually bad to discard
            return -0.15 if tricks_needed > 0 else 0.05

        if card.is_standard_suit() and card.number:
            if card.number >= 10:
                # High cards - bad to discard if need wins
                return -0.1 if tricks_needed > 0 else 0.05
            if card.number <= 5:
                # Low cards - good to discard
                return 0.1

        return 0.0

    def action_masks(self) -> np.ndarray:
        """Return valid action mask based on current decision phase."""
        # Delegate to parent for playing phase
        if self.decision_phase == DecisionPhase.PLAYING:
            return super().action_masks()

        mask = np.zeros(11, dtype=bool)

        if self.decision_phase == DecisionPhase.ABILITY_ROSIE:
            # All players are valid choices
            num_players = len(self.game.players) if self.game else 4
            mask[:min(num_players, 6)] = True

        elif self.decision_phase == DecisionPhase.ABILITY_BENDT_1:
            # All cards in hand are valid
            agent = self.game.get_player(self.agent_player_id) if self.game else None
            if agent and agent.hand:
                mask[:min(len(agent.hand), 11)] = True
            else:
                mask[0] = True

        elif self.decision_phase == DecisionPhase.ABILITY_BENDT_2:
            # All cards except first discard are valid
            agent = self.game.get_player(self.agent_player_id) if self.game else None
            if agent and agent.hand:
                for i, card_id in enumerate(agent.hand[:11]):
                    if card_id != self.bendt_first_discard:
                        mask[i] = True
            if not mask.any():
                mask[0] = True

        elif self.decision_phase in (DecisionPhase.ABILITY_ROATAN, DecisionPhase.ABILITY_HARRY):
            # 3 options for both Roatán (0/10/20) and Harry (-1/0/+1)
            mask[:3] = True

        # Ensure at least one valid action
        if not mask.any():
            mask[0] = True

        return mask

    def _extend_observation(self, base_obs: np.ndarray) -> np.ndarray:
        """Extend base observation with phase and ability context."""
        extended = np.zeros(self.OBS_DIM_EXTENDED, dtype=np.float32)

        # Copy base observation
        extended[: self.OBS_DIM] = base_obs

        # Add phase one-hot (6 dims)
        phase_idx = self.OBS_DIM
        extended[phase_idx + int(self.decision_phase)] = 1.0

        # Add ability context (34 dims)
        context_idx = phase_idx + self.DECISION_PHASE_DIM
        ability_context = self._build_ability_context()
        extended[context_idx : context_idx + self.ABILITY_CONTEXT_DIM] = ability_context

        return extended

    def _build_ability_context(self) -> np.ndarray:
        """Build ability-specific context for current phase."""
        context = np.zeros(self.ABILITY_CONTEXT_DIM, dtype=np.float32)

        if not self.game:
            return context

        agent = self.game.get_player(self.agent_player_id)
        current_round = self.game.get_current_round()

        # Rosie context (6 dims): player positions/lead strengths
        if self.decision_phase == DecisionPhase.ABILITY_ROSIE:
            for i, player in enumerate(self.game.players[:6]):
                # Estimate lead strength based on score and bid
                strength = 0.5
                if player.score:
                    strength = min(1.0, player.score / 200.0)
                context[i] = strength

        # Bendt context (20 dims): drawn cards + discard state
        elif self.decision_phase in (DecisionPhase.ABILITY_BENDT_1, DecisionPhase.ABILITY_BENDT_2):
            # Encode drawn cards (18 dims = 2 cards x 9 features)
            for i, card_id in enumerate(self.bendt_drawn_cards[:2]):
                card = get_card(card_id)
                offset = 6 + i * 9
                context[offset : offset + 9] = self._encode_card_for_context(card)

            # Discard progress (2 dims)
            context[24] = 1.0 if self.decision_phase == DecisionPhase.ABILITY_BENDT_2 else 0.0
            if self.bendt_first_discard:
                context[25] = 1.0

        # Roatán context (2 dims): bid confidence
        elif self.decision_phase == DecisionPhase.ABILITY_ROATAN:
            if current_round:
                tricks_needed = self.goal_bid - self.tricks_won
                tricks_remaining = self.current_round_num - len(current_round.tricks)
                context[26] = tricks_needed / max(tricks_remaining, 1)  # Need ratio
                context[27] = 1.0 if tricks_needed <= 0 else 0.0  # Already at goal

        # Harry context (4 dims): bid/tricks state
        elif self.decision_phase == DecisionPhase.ABILITY_HARRY and agent:
            bid = agent.bid or 0
            context[28] = bid / MAX_ROUNDS
            context[29] = self.tricks_won / MAX_ROUNDS
            context[30] = (self.tricks_won - bid) / MAX_ROUNDS  # Difference
            context[31] = 1.0 if self.tricks_won == bid else 0.0  # Exact match

        return context

    def _encode_card_for_context(self, card) -> np.ndarray:
        """Encode a card as 9 features for ability context."""
        features = np.zeros(9, dtype=np.float32)

        if card.is_escape():
            features[0] = 1.0
        elif card.is_pirate():
            features[1] = 1.0
        elif card.is_king():
            features[2] = 1.0
        elif card.is_mermaid():
            features[3] = 1.0
        elif card.is_tigress():
            features[4] = 1.0
        elif card.is_standard_suit():
            features[5] = 1.0
            if card.number:
                features[6] = card.number / 14.0
        elif card.is_loot():
            features[7] = 1.0

        features[8] = self._card_strength(card)
        return features

    def _get_extended_obs(self) -> np.ndarray:
        """Get extended observation including phase and ability context."""
        base_obs = self._get_worker_obs()
        return self._extend_observation(base_obs)

    def set_abilities_enabled(self, enabled: bool) -> None:
        """Enable or disable ability decisions (for curriculum)."""
        self.enable_abilities = enabled


def create_ability_env(
    opponent_type: str = "rule_based",
    difficulty: str = "medium",
    fixed_goal: int | None = None,
    enable_abilities: bool = True,
):
    """Create ability-aware environment with action masking."""
    env = AbilityAwareEnv(
        opponent_bot_type=opponent_type,
        opponent_difficulty=difficulty,
        fixed_goal=fixed_goal,
        enable_abilities=enable_abilities,
    )

    def mask_fn(env: AbilityAwareEnv) -> np.ndarray:
        return env.action_masks()

    return ActionMasker(env, mask_fn)
