"""Logging service."""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class LogService:
    """
    Service for structured logging.

    Provides consistent logging across the application.
    """

    def info(self, data: Dict[str, str]) -> None:
        """
        Log info message.

        Args:
            data: Log data as key-value pairs
        """
        message = " | ".join(f"{k}={v}" for k, v in data.items())
        logger.info(message)

    def error(self, data: Dict[str, str]) -> None:
        """
        Log error message.

        Args:
            data: Log data as key-value pairs
        """
        message = " | ".join(f"{k}={v}" for k, v in data.items())
        logger.error(message)

    def warning(self, data: Dict[str, str]) -> None:
        """
        Log warning message.

        Args:
            data: Log data as key-value pairs
        """
        message = " | ".join(f"{k}={v}" for k, v in data.items())
        logger.warning(message)

    def debug(self, data: Dict[str, str]) -> None:
        """
        Log debug message.

        Args:
            data: Log data as key-value pairs
        """
        message = " | ".join(f"{k}={v}" for k, v in data.items())
        logger.debug(message)
