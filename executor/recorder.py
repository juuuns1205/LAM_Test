"""Skeleton script for recording user interactions in Selenium.

This module sets up a Chrome WebDriver instance (managed via
``webdriver-manager``) that listens for click and text entry events.
Captured interactions are stored in ``data/actions.json`` following the
``{"action": str, "selector": str, "value": str | None, "timestamp": float}``
schema.

The implementation focuses on the plumbing necessary to capture events and
persist them in a JSON array. Application-specific behavior (for example,
what URL to load or when to stop recording) can be added by importing this
module and customizing the ``main`` function or building on top of
``create_recording_driver``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.events import (
    AbstractEventListener,
    EventFiringWebDriver,
)
from webdriver_manager.chrome import ChromeDriverManager


DATA_DIR = Path("data")
ACTION_LOG_PATH = DATA_DIR / "actions.json"


@dataclass
class RecordedAction:
    """Represents a single recorded user interaction."""

    action: str
    selector: str
    value: Optional[str]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "selector": self.selector,
            "value": self.value,
            "timestamp": self.timestamp,
        }


class ActionEventListener(AbstractEventListener):
    """Event listener that captures click and text entry events."""

    def __init__(self, output_path: Path):
        super().__init__()
        self.output_path = output_path
        self._initialise_output_file()

    # ------------------------------------------------------------------
    # Selenium event hooks
    # ------------------------------------------------------------------
    def after_click(self, element: WebElement, driver: webdriver.Chrome) -> None:
        action = RecordedAction(
            action="click",
            selector=self._css_selector_for(element),
            value=None,
            timestamp=time.time(),
        )
        self._append_action(action)

    def after_change_value_of(
        self, element: WebElement, driver: webdriver.Chrome
    ) -> None:
        action = RecordedAction(
            action="input",
            selector=self._css_selector_for(element),
            value=element.get_attribute("value"),
            timestamp=time.time(),
        )
        self._append_action(action)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialise_output_file(self) -> None:
        """Ensure the output file exists and contains a JSON array."""

        os.makedirs(self.output_path.parent, exist_ok=True)
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            self.output_path.write_text("[]", encoding="utf-8")

    def _append_action(self, action: RecordedAction) -> None:
        """Append a recorded action to the JSON log file."""

        try:
            with self.output_path.open("r", encoding="utf-8") as fp:
                actions: list[Dict[str, Any]] = json.load(fp)
        except json.JSONDecodeError:
            actions = []

        actions.append(action.to_dict())

        with self.output_path.open("w", encoding="utf-8") as fp:
            json.dump(actions, fp, indent=2, ensure_ascii=False)

    @staticmethod
    def _css_selector_for(element: WebElement) -> str:
        """Attempt to build a stable CSS selector for a web element."""

        element_id = element.get_attribute("id")
        if element_id:
            return f"#{element_id}"

        name = element.get_attribute("name")
        if name:
            return f"[name='{name}']"

        classes = element.get_attribute("class")
        if classes:
            class_selector = ".".join(
                cls for cls in classes.split() if cls.strip()
            )
            if class_selector:
                return f"{element.tag_name}.{class_selector}"

        return element.tag_name


def create_recording_driver(headless: bool = False) -> EventFiringWebDriver:
    """Create a Chrome WebDriver wrapped with the action recorder.

    Parameters
    ----------
    headless:
        Whether to launch Chrome in headless mode.
    """

    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    listener = ActionEventListener(ACTION_LOG_PATH)
    return EventFiringWebDriver(driver, listener)


def main() -> None:
    """Example entry point for manual testing.

    Update this function to navigate to the desired URL and orchestrate user
    interactions. The driver returned from ``create_recording_driver`` will
    automatically log click and input events to ``data/actions.json``.
    """

    driver = create_recording_driver()
    try:
        driver.get("https://example.com")
        input("Press Enter to stop recording...")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()

