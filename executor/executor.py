"""Playback Selenium actions recorded to JSON.

This module reads a sequence of recorded actions from a JSON file and
reproduces them in a Chrome browser driven by Selenium.  The JSON format is
expected to follow the schema described in the project README.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


LOGGER = logging.getLogger(__name__)


@dataclass
class Action:
    """Represents a single user interaction to be replayed."""

    kind: str
    xpath: str
    timestamp: Optional[float]
    value: Optional[str]

    SUPPORTED_ACTIONS = {"click", "input"}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Action":
        """Create an :class:`Action` from a JSON dictionary.

        Parameters
        ----------
        raw:
            Dictionary loaded from the JSON file.

        Returns
        -------
        Action
            Parsed action that can be executed by Selenium.

        Raises
        ------
        ValueError
            If the dictionary does not conform to the expected schema.
        """

        action_type = raw.get("action")
        if action_type not in cls.SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported action type: {action_type!r}")

        selector = raw.get("selector") or {}
        xpath = selector.get("xpath")
        if not xpath:
            raise ValueError("Each action must provide selector['xpath'].")

        timestamp = raw.get("timestamp")
        if timestamp is not None:
            try:
                timestamp = float(timestamp)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError("timestamp must be numeric or null") from exc

        value = raw.get("value")
        if action_type == "input" and value is None:
            raise ValueError("Input actions require a 'value' field.")

        return cls(action_type, xpath, timestamp, value)

    def perform(self, driver: webdriver.Chrome) -> None:
        """Execute this action using the provided Selenium driver."""

        element = driver.find_element(By.XPATH, self.xpath)
        if self.kind == "click":
            element.click()
        elif self.kind == "input":
            element.clear()
            element.send_keys(self.value)
        else:  # pragma: no cover - should be unreachable
            raise RuntimeError(f"Unsupported action type: {self.kind}")


def load_actions(path: Path) -> List[Action]:
    """Load actions from the given JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("actions.json must contain a list of actions.")

    return [Action.from_dict(item) for item in data]


def build_driver(headless: bool = True) -> webdriver.Chrome:
    """Create a Chrome WebDriver instance using webdriver-manager."""

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def execute_actions(actions: List[Action], driver: webdriver.Chrome) -> None:
    """Execute a list of actions with timestamp-aware delays."""

    previous_timestamp: Optional[float] = None

    for index, action in enumerate(actions):
        if index > 0:
            current_timestamp = action.timestamp
            if (
                previous_timestamp is not None
                and current_timestamp is not None
                and current_timestamp > previous_timestamp
            ):
                delay_seconds = (current_timestamp - previous_timestamp) / 1000.0
                LOGGER.debug("Sleeping for %.3f seconds before next action", delay_seconds)
                time.sleep(delay_seconds)
        LOGGER.info("Executing %s action on %s", action.kind, action.xpath)
        action.perform(driver)
        if action.timestamp is not None:
            previous_timestamp = action.timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Selenium actions from JSON")
    default_actions_path = Path(__file__).resolve().parents[1] / "data" / "actions.json"
    parser.add_argument(
        "--actions",
        type=Path,
        default=default_actions_path,
        help="Path to the actions JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run Chrome with a visible window instead of headless mode.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., DEBUG, INFO, WARNING)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    actions_path = args.actions
    if not actions_path.exists():
        raise FileNotFoundError(f"Action file not found: {actions_path}")

    actions = load_actions(actions_path)
    LOGGER.info("Loaded %d actions from %s", len(actions), actions_path)

    driver = build_driver(headless=not args.no_headless)
    try:
        execute_actions(actions, driver)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
