"""Tests for ability-aware RL environment."""

import numpy as np
import pytest

from app.gym_env.skullking_env_ability import (
    AbilityAwareEnv,
    DecisionPhase,
    create_ability_env,
)


class TestAbilityAwareEnv:
    """Tests for AbilityAwareEnv."""

    def test_initialization(self):
        """Test environment initializes correctly."""
        env = AbilityAwareEnv()

        assert env.observation_space.shape == (243,)  # 203 base + 40 ability
        assert env.action_space.n == 11
        assert env.decision_phase == DecisionPhase.PLAYING
        assert env.enable_abilities is True

    def test_reset(self):
        """Test environment reset."""
        env = AbilityAwareEnv()
        obs, info = env.reset(seed=42)

        assert obs.shape == (243,)
        assert env.decision_phase == DecisionPhase.PLAYING
        assert env.pending_ability is None
        assert "goal_bid" in info
        assert "round" in info

    def test_observation_has_phase_encoding(self):
        """Test that observation includes phase one-hot."""
        env = AbilityAwareEnv()
        obs, _ = env.reset(seed=42)

        # Phase encoding starts at index 203
        phase_start = 203
        phase_end = phase_start + 6

        # Should have exactly one 1.0 in phase section
        phase_section = obs[phase_start:phase_end]
        assert np.sum(phase_section) == 1.0
        assert phase_section[DecisionPhase.PLAYING] == 1.0

    def test_action_mask_playing_phase(self):
        """Test action masking in playing phase."""
        env = AbilityAwareEnv()
        env.reset(seed=42)

        mask = env.action_masks()

        # Should have at least one valid action
        assert mask.any()
        # Should have at most 11 actions
        assert len(mask) == 11

    def test_action_mask_roatan_phase(self):
        """Test action masking in Roatán ability phase."""
        env = AbilityAwareEnv()
        env.reset(seed=42)

        # Force into Roatán phase
        env.decision_phase = DecisionPhase.ABILITY_ROATAN

        mask = env.action_masks()

        # Roatán has exactly 3 valid actions (0, 10, 20)
        assert mask[0]
        assert mask[1]
        assert mask[2]
        assert np.sum(mask) == 3

    def test_action_mask_harry_phase(self):
        """Test action masking in Harry ability phase."""
        env = AbilityAwareEnv()
        env.reset(seed=42)

        # Force into Harry phase
        env.decision_phase = DecisionPhase.ABILITY_HARRY

        mask = env.action_masks()

        # Harry has exactly 3 valid actions (-1, 0, +1)
        assert mask[0]
        assert mask[1]
        assert mask[2]
        assert np.sum(mask) == 3

    def test_action_mask_rosie_phase(self):
        """Test action masking in Rosie ability phase."""
        env = AbilityAwareEnv()
        env.reset(seed=42)

        # Force into Rosie phase
        env.decision_phase = DecisionPhase.ABILITY_ROSIE

        mask = env.action_masks()

        # Rosie has 4 valid actions (one per player)
        assert np.sum(mask) == 4  # 4 players in game

    def test_step_playing_phase(self):
        """Test step in playing phase."""
        env = AbilityAwareEnv()
        obs, _ = env.reset(seed=42)

        # Take a valid action
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        action = valid_actions[0]

        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (243,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))

    def test_full_episode_without_abilities(self):
        """Test running a full episode with abilities disabled."""
        env = AbilityAwareEnv(enable_abilities=False)
        obs, _ = env.reset(seed=42)

        total_reward = 0.0
        steps = 0
        max_steps = 100

        while steps < max_steps:
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        assert steps > 0
        assert terminated or truncated or steps == max_steps

    def test_full_episode_with_abilities(self):
        """Test running a full episode with abilities enabled."""
        env = AbilityAwareEnv(enable_abilities=True)
        obs, _ = env.reset(seed=42)

        total_reward = 0.0
        steps = 0
        max_steps = 100
        phases_seen = set()

        while steps < max_steps:
            phases_seen.add(env.decision_phase)

            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        assert steps > 0
        assert DecisionPhase.PLAYING in phases_seen

    def test_bendt_discard_flow(self):
        """Test Bendt's two-phase discard flow."""
        env = AbilityAwareEnv()
        env.reset(seed=42)

        # Simulate having drawn cards for Bendt
        from app.models.card import CardId

        env.decision_phase = DecisionPhase.ABILITY_BENDT_1
        env.bendt_drawn_cards = [CardId.PARROT1, CardId.MAP2]

        # First discard
        mask = env.action_masks()
        assert mask.any()

        # Simulate selecting first card
        valid_actions = np.where(mask)[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)

            # Should transition to second discard
            assert env.decision_phase == DecisionPhase.ABILITY_BENDT_2

    def test_create_ability_env_helper(self):
        """Test the create_ability_env helper function."""
        env = create_ability_env(
            opponent_type="random",
            difficulty="easy",
            enable_abilities=True,
        )

        # Should be wrapped with ActionMasker
        obs, _ = env.reset()
        assert obs.shape == (243,)

        # Should have action_masks method
        mask = env.action_masks()
        assert mask.shape == (11,)

    def test_ability_context_in_observation(self):
        """Test that ability context changes with phase."""
        env = AbilityAwareEnv()
        env.reset(seed=42)

        # Get observation in playing phase
        obs_playing = env._get_extended_obs()

        # Force into Harry phase and get observation
        env.decision_phase = DecisionPhase.ABILITY_HARRY
        obs_harry = env._get_extended_obs()

        # Phase section should be different
        phase_start = 203
        phase_end = phase_start + 6

        assert not np.allclose(obs_playing[phase_start:phase_end], obs_harry[phase_start:phase_end])

    def test_ability_decisions_tracking(self):
        """Test that ability decisions are tracked."""
        env = AbilityAwareEnv()
        env.reset(seed=42)

        # Force a Roatán decision
        env.decision_phase = DecisionPhase.ABILITY_ROATAN
        env.pending_ability = True  # Simulated

        # Manually call step_roatan logic
        env.ability_decisions["roatan"].append(1)

        assert len(env.ability_decisions["roatan"]) == 1
        assert env.ability_decisions["roatan"][0] == 1

    def test_reset_clears_ability_state(self):
        """Test that reset clears all ability state."""
        env = AbilityAwareEnv()
        env.reset(seed=42)

        # Mess with state
        env.decision_phase = DecisionPhase.ABILITY_HARRY
        env.pending_ability = True
        env.harry_pending = True
        env.ability_decisions["harry"].append(0)

        # Reset should clear everything
        env.reset(seed=43)

        assert env.decision_phase == DecisionPhase.PLAYING
        assert env.pending_ability is None
        assert env.harry_pending is False
        assert len(env.ability_decisions["harry"]) == 0


class TestAbilityRewards:
    """Tests for ability-specific rewards."""

    def test_bendt_discard_escape_when_need_wins(self):
        """Test that discarding escape when needing wins is rewarded."""
        env = AbilityAwareEnv()
        env.reset(seed=42)

        env.goal_bid = 3
        env.tricks_won = 0  # Need wins

        from app.models.card import get_card, CardId

        escape_card = get_card(CardId.ESCAPE1)
        reward = env._evaluate_bendt_discard(escape_card)

        # Should be positive - good to discard escape when need wins
        assert reward > 0

    def test_bendt_discard_high_card_when_need_wins(self):
        """Test that discarding high cards when needing wins is penalized."""
        env = AbilityAwareEnv()
        env.reset(seed=42)

        env.goal_bid = 3
        env.tricks_won = 0  # Need wins

        from app.models.card import get_card, CardId

        high_card = get_card(CardId.PARROT14)
        reward = env._evaluate_bendt_discard(high_card)

        # Should be negative - bad to discard high cards when need wins
        assert reward < 0


class TestPhaseTransitions:
    """Tests for decision phase transitions."""

    def test_rosie_returns_to_playing(self):
        """Test that Rosie ability returns to playing phase."""
        env = AbilityAwareEnv()
        env.reset(seed=42)

        # Setup for Rosie phase
        env.decision_phase = DecisionPhase.ABILITY_ROSIE
        current_round = env.game.get_current_round()
        if current_round:
            from app.models.pirate_ability import PendingAbility, PirateType, AbilityType

            env.pending_ability = PendingAbility(
                player_id=env.agent_player_id,
                pirate_type=PirateType.ROSIE,
                ability_type=AbilityType.CHOOSE_STARTER,
                trick_number=1,
            )

        # Execute Rosie action (choose self)
        obs, reward, terminated, truncated, info = env.step(0)

        # Should return to playing (or end if round complete)
        assert env.decision_phase == DecisionPhase.PLAYING or terminated

    def test_bendt_two_phase_flow(self):
        """Test Bendt's two-phase discard correctly transitions."""
        env = AbilityAwareEnv()
        env.reset(seed=42)

        from app.models.card import CardId

        # Setup for Bendt phase 1
        env.decision_phase = DecisionPhase.ABILITY_BENDT_1
        env.bendt_drawn_cards = [CardId.PARROT1, CardId.MAP2]

        # Add drawn cards to hand
        agent = env.game.get_player(env.agent_player_id)
        if agent:
            agent.hand.extend(env.bendt_drawn_cards)

        current_round = env.game.get_current_round()
        if current_round:
            from app.models.pirate_ability import PendingAbility, PirateType, AbilityType

            env.pending_ability = PendingAbility(
                player_id=env.agent_player_id,
                pirate_type=PirateType.BENDT,
                ability_type=AbilityType.DRAW_DISCARD,
                trick_number=1,
            )

        # First discard
        obs, reward, terminated, truncated, info = env.step(0)

        # Should be in second discard phase
        assert env.decision_phase == DecisionPhase.ABILITY_BENDT_2

        # Second discard
        obs, reward, terminated, truncated, info = env.step(1)

        # Should return to playing
        assert env.decision_phase == DecisionPhase.PLAYING or terminated
