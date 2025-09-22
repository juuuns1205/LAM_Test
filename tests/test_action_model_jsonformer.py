"""Tests for the Jsonformer-backed action model utilities."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

action_model = importlib.import_module("lam.action_model")


def test_run_local_model_uses_jsonformer_and_returns_structured_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the Jsonformer integration yields structured data for downstream parsing."""

    sample_prompt = "로그인 버튼 클릭"
    expected_actions: List[Dict[str, Any]] = [
        {
            "action": "click",
            "selector": {"css": ".login", "xpath": None},
            "value": None,
            "timestamp": None,
        }
    ]

    captured: Dict[str, Any] = {}

    class DummyJsonformer:
        def __init__(self, model: Any, tokenizer: Any, schema: Dict[str, Any], prompt: str) -> None:
            captured["model"] = model
            captured["tokenizer"] = tokenizer
            captured["schema"] = schema
            captured["prompt"] = prompt

        def __call__(self) -> List[Dict[str, Any]]:
            return [
                {
                    "action": "click",
                    "selector": {"css": ".login", "xpath": None},
                    "value": None,
                    "timestamp": None,
                }
            ]

    monkeypatch.setattr(action_model, "torch", object())
    monkeypatch.setattr(action_model, "Jsonformer", DummyJsonformer)
    monkeypatch.setattr(
        action_model,
        "_load_or_initialise_model",
        lambda: (object(), object(), "cpu"),
    )

    actions = action_model._convert_step_to_actions(sample_prompt)

    assert actions == expected_actions
    assert captured["prompt"] == sample_prompt

    schema = captured["schema"]
    assert schema["anyOf"][0]["properties"]["action"]["enum"] == sorted(action_model._SUPPORTED_ACTIONS)
    assert schema["anyOf"][0]["additionalProperties"] is False
    assert schema["anyOf"][1]["type"] == "array"
