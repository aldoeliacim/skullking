"""Pytest configuration for API tests."""

import pytest


@pytest.fixture
def anyio_backend():
    """Configure anyio to use asyncio backend."""
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_event_recorder():
    """Reset the event recorder between tests to avoid state pollution."""
    from app.services.event_recorder import event_recorder

    # Clear all state before each test
    event_recorder._events.clear()
    event_recorder._game_start_times.clear()
    event_recorder._histories.clear()

    yield

    # Clear after test as well
    event_recorder._events.clear()
    event_recorder._game_start_times.clear()
    event_recorder._histories.clear()
