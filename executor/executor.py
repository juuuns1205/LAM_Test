"""Playback Selenium actions recorded to JSON.

Reads a sequence of recorded actions from a JSON file and
reproduces them in a Chrome browser driven by Selenium.
"""

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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


LOGGER = logging.getLogger(__name__)

# ✅ 여기서 직접 URL 지정
START_URL = "https://www.naver.com"

# ✅ actions.json 경로 지정
ACTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "actions.json"


@dataclass
class Action:
    kind: str
    xpath: Optional[str]
    css: Optional[str]
    timestamp: Optional[float]
    value: Optional[str]

    SUPPORTED_ACTIONS = {"click", "input"}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Action":
        action_type = raw.get("action")
        if action_type not in cls.SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported action type: {action_type!r}")

        selector = raw.get("selector") or {}
        xpath = selector.get("xpath")
        css = selector.get("css")

        timestamp = raw.get("timestamp")
        if timestamp is not None:
            timestamp = float(timestamp)

        value = raw.get("value")
        if action_type == "input" and value is None:
            raise ValueError("Input actions require a 'value' field.")

        return cls(action_type, xpath, css, timestamp, value)

    def perform(self, driver: webdriver.Chrome) -> None:
        wait = WebDriverWait(driver, 10)
        element = None

        if self.xpath:
            try:
                element = wait.until(EC.presence_of_element_located((By.XPATH, self.xpath)))
            except Exception:
                LOGGER.warning("XPath failed: %s", self.xpath)

        if element is None and self.css:
            element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, self.css)))

        if element is None:
            raise RuntimeError(f"Could not locate element: {self.xpath or self.css}")

        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)

        if self.kind == "click":
            element.click()
        elif self.kind == "input":
            element.clear()
            element.send_keys(self.value)


def load_actions(path: Path) -> List[Action]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("actions.json must contain a list of actions.")

    return [Action.from_dict(item) for item in data]


def build_driver(headless: bool = False) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def execute_actions(actions: List[Action], driver: webdriver.Chrome) -> None:
    previous_timestamp: Optional[float] = None

    for index, action in enumerate(actions):
        if index > 0 and previous_timestamp is not None and action.timestamp is not None:
            delay_seconds = (action.timestamp - previous_timestamp) / 1000.0
            time.sleep(min(max(delay_seconds, 0), 2.0))  # 최대 2초로 제한

        LOGGER.info("Executing %s action (%s)", action.kind, action.xpath or action.css)
        action.perform(driver)

        if action.timestamp is not None:
            previous_timestamp = action.timestamp


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    if not ACTIONS_PATH.exists():
        raise FileNotFoundError(f"Action file not found: {ACTIONS_PATH}")

    actions = load_actions(ACTIONS_PATH)
    LOGGER.info("Loaded %d actions from %s", len(actions), ACTIONS_PATH)

    driver = build_driver(headless=False)
    driver.get(START_URL)  # ✅ URL 직접 변수로 지정
    try:
        execute_actions(actions, driver)
        input("Press Enter to quit...")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
