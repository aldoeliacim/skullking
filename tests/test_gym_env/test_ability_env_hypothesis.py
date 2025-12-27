"""Property-based tests for ability-aware RL environment using Hypothesis.

These tests generate random inputs to find edge cases that unit tests might miss.
For example, this would catch bugs like the PHASE_DIM collision where observation
space dimensions were incorrect.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.gym_env.skullking_env_ability import (
    AbilityAwareEnv,
    DecisionPhase,
)


class TestAbilityEnvProperties:
    """Property-based tests for AbilityAwareEnv invariants."""

    @given(seed=st.integers(0, 100000))
    @settings(max_examples=50, deadline=None)
    def test_reset_returns_correct_shape(self, seed: int) -> None:
        """Reset always returns observation with correct shape."""
        env = AbilityAwareEnv()
        obs, info = env.reset(seed=seed)

        assert obs.shape == (243,), f"Expected (243,), got {obs.shape}"
        assert isinstance(info, dict)

    @given(seed=st.integers(0, 100000))
    @settings(max_examples=50, deadline=None)
    def test_observation_bounds(self, seed: int) -> None:
        """Observations should be within reasonable bounds."""
        env = AbilityAwareEnv()
        obs, _ = env.reset(seed=seed)

        # All values should be finite
        assert np.all(np.isfinite(obs)), "Observation contains non-finite values"

        # One-hot encodings should be 0 or 1
        # Phase encoding is at indices 203:209
        phase_section = obs[203:209]
        assert np.all((phase_section == 0) | (phase_section == 1)), "Phase encoding not one-hot"
        assert np.sum(phase_section) == 1, "Phase encoding should have exactly one 1"

    @given(seed=st.integers(0, 100000))
    @settings(max_examples=30, deadline=None)
    def test_action_mask_valid(self, seed: int) -> None:
        """Action mask should always have at least one valid action."""
        env = AbilityAwareEnv()
        env.reset(seed=seed)

        mask = env.action_masks()

        assert mask.shape == (11,), f"Expected mask shape (11,), got {mask.shape}"
        assert mask.dtype == np.bool_, f"Expected bool dtype, got {mask.dtype}"
        assert np.any(mask), "Action mask has no valid actions"

    @given(seed=st.integers(0, 100000), steps=st.integers(1, 20))
    @settings(max_examples=30, deadline=None)
    def test_step_maintains_invariants(self, seed: int, steps: int) -> None:
        """Step maintains observation shape and returns valid types."""
        env = AbilityAwareEnv()
        obs, _ = env.reset(seed=seed)

        for _ in range(steps):
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            if len(valid_actions) == 0:
                break

            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)

            # Check types
            assert obs.shape == (243,), f"Obs shape changed to {obs.shape}"
            assert isinstance(reward, (float, np.floating))
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert isinstance(info, dict)

            if terminated or truncated:
                break

    @given(seed=st.integers(0, 100000))
    @settings(max_examples=20, deadline=None)
    def test_phase_encoding_consistency(self, seed: int) -> None:
        """Phase in observation matches decision_phase attribute."""
        env = AbilityAwareEnv()
        env.reset(seed=seed)

        obs = env._get_extended_obs()
        phase_section = obs[203:209]

        # Find which phase is encoded
        encoded_phase = np.argmax(phase_section)
        assert encoded_phase == env.decision_phase.value

    @given(seed=st.integers(0, 100000))
    @settings(max_examples=30, deadline=None)
    def test_invalid_action_rejected(self, seed: int) -> None:
        """Invalid actions should not crash the environment."""
        env = AbilityAwareEnv()
        env.reset(seed=seed)

        mask = env.action_masks()
        invalid_actions = np.where(~mask)[0]

        if len(invalid_actions) > 0:
            # Taking an invalid action should either raise or return penalty
            # The environment should not crash
            try:
                obs, reward, terminated, truncated, info = env.step(invalid_actions[0])
                # If it doesn't raise, observation should still be valid
                assert obs.shape == (243,)
            except (ValueError, IndexError):
                # Expected - invalid action was rejected
                pass

    @given(
        seed=st.integers(0, 100000),
        phase=st.sampled_from(list(DecisionPhase)),
    )
    @settings(max_examples=30, deadline=None)
    def test_forced_phase_action_mask(self, seed: int, phase: DecisionPhase) -> None:
        """Action mask should be valid for any forced phase."""
        env = AbilityAwareEnv()
        env.reset(seed=seed)

        # Force into specific phase
        env.decision_phase = phase

        mask = env.action_masks()

        assert mask.shape == (11,)
        assert mask.dtype == np.bool_

        # Verify phase-specific constraints
        if phase == DecisionPhase.ABILITY_ROATAN:
            # Roatán has exactly 3 valid actions
            assert np.sum(mask) == 3
            assert mask[0] and mask[1] and mask[2]
        elif phase == DecisionPhase.ABILITY_HARRY:
            # Harry has exactly 3 valid actions
            assert np.sum(mask) == 3
            assert mask[0] and mask[1] and mask[2]


class TestAbilityEnvStress:
    """Stress tests for AbilityAwareEnv."""

    @given(seed=st.integers(0, 100000))
    @settings(max_examples=10, deadline=None)
    def test_full_episode_completion(self, seed: int) -> None:
        """Environment should complete episodes without crashing."""
        env = AbilityAwareEnv(enable_abilities=True)
        obs, _ = env.reset(seed=seed)

        steps = 0
        max_steps = 200

        while steps < max_steps:
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]

            if len(valid_actions) == 0:
                pytest.fail("No valid actions available mid-episode")

            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            if terminated or truncated:
                break

        # Should complete within max_steps
        assert terminated or truncated or steps == max_steps

    @given(seed=st.integers(0, 100000))
    @settings(max_examples=10, deadline=None)
    def test_multiple_resets(self, seed: int) -> None:
        """Multiple resets should not cause state leakage."""
        env = AbilityAwareEnv()

        for i in range(5):
            obs, info = env.reset(seed=seed + i)

            assert obs.shape == (243,)
            assert env.decision_phase == DecisionPhase.PLAYING
            assert env.pending_ability is None

            # Take a few steps
            for _ in range(3):
                mask = env.action_masks()
                valid_actions = np.where(mask)[0]
                if len(valid_actions) == 0:
                    break
                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break


class TestObservationSpaceConsistency:
    """Tests for observation space consistency across parent/child classes."""

    def test_observation_dimensions_match_space(self) -> None:
        """Observation dimensions should match declared space."""
        env = AbilityAwareEnv()
        obs, _ = env.reset(seed=42)

        expected_shape = env.observation_space.shape
        assert obs.shape == expected_shape, (
            f"Observation shape {obs.shape} doesn't match observation_space.shape {expected_shape}"
        )

    def test_parent_dimensions_not_overwritten(self) -> None:
        """Child class should not accidentally override parent dimensions."""
        env = AbilityAwareEnv()

        # These are the expected dimensions
        # Parent WorkerEnv uses PHASE_DIM = 3 for its internal phase
        # Child uses DECISION_PHASE_DIM = 6 for ability phases
        assert hasattr(env, "DECISION_PHASE_DIM")
        assert env.DECISION_PHASE_DIM == 6

        # The observation should be:
        # 203 (base WorkerEnv) + 6 (decision phase) + 34 (ability context) = 243
        assert env.observation_space.shape == (243,)

    @given(seed=st.integers(0, 10000))
    @settings(max_examples=20, deadline=None)
    def test_extended_obs_indices_valid(self, seed: int) -> None:
        """Extended observation should have valid indices."""
        env = AbilityAwareEnv()
        env.reset(seed=seed)

        obs = env._get_extended_obs()

        # Base observation (inherited from parent)
        base_end = 203

        # Phase encoding
        phase_start = base_end
        phase_end = phase_start + 6

        # Ability context
        context_start = phase_end
        context_end = context_start + 34

        assert context_end == 243, f"Expected 243, got {context_end}"

        # All sections should be accessible
        _ = obs[:base_end]  # Base obs
        _ = obs[phase_start:phase_end]  # Phase
        _ = obs[context_start:context_end]  # Context
