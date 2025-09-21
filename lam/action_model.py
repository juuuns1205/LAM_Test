"""Utility module to convert plan steps into executable Selenium actions."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_SUPPORTED_ACTIONS = {"click", "input"}


def get_actions(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a plan dictionary into a list of Selenium action dictionaries."""

    if not isinstance(plan, dict):
        raise ValueError("plan must be provided as a dictionary containing a 'plan' list")

    steps = plan.get("plan", [])
    if not isinstance(steps, list):
        raise ValueError("plan['plan'] must be a list of step definitions")

    aggregated_actions: List[Dict[str, Any]] = []
    for index, step in enumerate(steps):
        if isinstance(step, dict):
            step_text = step.get("step")
        else:
            step_text = step

        if not isinstance(step_text, str) or not step_text.strip():
            logger.warning("Skipping step %s because it does not contain valid text: %r", index, step)
            continue

        actions_for_step = _convert_step_to_actions(step_text.strip())
        aggregated_actions.extend(actions_for_step)

    return aggregated_actions


def _convert_step_to_actions(step_text: str) -> List[Dict[str, Any]]:
    """Convert a single natural language step description to Selenium actions."""

    try:
        model_output = run_local_model(step_text)
        if not model_output:
            raise ValueError("Model did not return any output")

        parsed_output = json.loads(model_output)
    except Exception as exc:  # pragma: no cover - defensive error handling
        logger.exception("Falling back to heuristic action for step %r due to model error", step_text)
        return [_fallback_action(step_text)]

    if isinstance(parsed_output, dict):
        parsed_actions = [parsed_output]
    elif isinstance(parsed_output, list):
        parsed_actions = [action for action in parsed_output if isinstance(action, dict)]
    else:
        logger.warning(
            "Model output for step %r is not a dict or list; falling back to heuristic action", step_text
        )
        return [_fallback_action(step_text)]

    normalised_actions = [_normalise_action(action) for action in parsed_actions]
    normalised_actions = [action for action in normalised_actions if action is not None]

    if not normalised_actions:
        logger.warning("Model did not produce valid actions for step %r; using fallback", step_text)
        return [_fallback_action(step_text)]

    return normalised_actions


def _normalise_action(action: Dict[str, Any]) -> Dict[str, Any] | None:
    """Normalise a model-produced action into the expected schema."""

    action_type = action.get("action")
    if action_type not in _SUPPORTED_ACTIONS:
        logger.debug("Unsupported action type %r received; defaulting to 'click'", action_type)
        action_type = "click"

    selector_data = action.get("selector")
    css_selector = None
    xpath_selector = None
    if isinstance(selector_data, dict):
        css_selector = selector_data.get("css")
        xpath_selector = selector_data.get("xpath")

    value = action.get("value")
    if value is not None and not isinstance(value, str):
        value = str(value)

    timestamp = action.get("timestamp") if action.get("timestamp") is not None else None

    return {
        "action": action_type,
        "selector": {"css": css_selector, "xpath": xpath_selector},
        "value": value,
        "timestamp": timestamp,
    }


def _fallback_action(step_text: str) -> Dict[str, Any]:
    """Return a simple click action targeting a button with the step text."""

    return {
        "action": "click",
        "selector": {"css": f"BUTTON[title='{step_text}']", "xpath": None},
        "value": None,
        "timestamp": None,
    }


def run_local_model(prompt: str) -> str:
    """Placeholder for invoking a local AI model to convert text into actions."""

    # TODO: 실제 로컬 모델 로직 (예: HuggingFace pipeline, PyTorch 모델 등)
    return json.dumps(
        [
            {
                "action": "click",
                "selector": {"css": "div.placeholder", "xpath": None},
                "value": None,
                "timestamp": None,
            }
        ]
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_plan = {
        "plan": [
            {"step": "로그인 페이지 열기"},
            {"step": "아이디 입력"},
            {"step": "로그인 버튼 클릭"},
        ]
    }

    actions = get_actions(sample_plan)
    print(json.dumps(actions, ensure_ascii=False, indent=2))
