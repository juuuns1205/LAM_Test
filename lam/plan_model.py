"""Utility module that converts natural language requests into executable plans.

This module provides a small wrapper around the OpenAI Responses API that turns a
user's natural language request into a structured step-by-step plan.  The module
also exposes a ``get_plan`` helper that can be reused across the project and
contains a fall-back implementation for environments where the API key is not
available.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv

try:  # pragma: no cover - defensive import guard
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled during runtime without OpenAI
    OpenAI = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """Represents a single step within a plan."""

    step: str

    def to_dict(self) -> Dict[str, str]:
        return {"step": self.step}


class PlanModelError(RuntimeError):
    """Raised when a plan cannot be generated via the OpenAI API."""


DEFAULT_MODEL = "gpt-4o-mini"
_SYSTEM_PROMPT = (
    "당신은 사용자의 자연어 명령을 기반으로 사람이 이해하기 쉬운 단계별 작업 계획을 "
    "생성하는 도우미입니다. 반드시 JSON 포맷으로만 응답하고, 다음 예시와 같은 구조를 따르세요:\n"
    "{\n  \"plan\": [\n    {\"step\": \"첫 번째 단계 설명\"},\n    {\"step\": \"두 번째 단계 설명\"}\n  ]\n}\n"
    "각 단계는 간결하지만 실행 가능한 명령문으로 작성하세요. 최소한 하나의 단계를 포함하고, "
    "사용자의 요청을 완료하기 위해 필요한 주요 과정을 모두 나열하세요."
)


def _load_client(api_key: Optional[str]) -> OpenAI:
    """Instantiate the OpenAI client, validating the availability of the SDK."""

    if OpenAI is None:
        raise PlanModelError(
            "The `openai` package is not available. Install the dependency to use the plan model."
        )

    if not api_key:
        raise PlanModelError(
            "OPENAI_API_KEY is not configured. Please provide the key in your environment or .env file."
        )

    return OpenAI(api_key=api_key)


def _call_openai_for_plan(client: OpenAI, user_query: str, model: str = DEFAULT_MODEL) -> str:
    """Call the OpenAI Responses API and return the raw JSON string provided by the model."""

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": _SYSTEM_PROMPT},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query},
                    ],
                },
            ],
            temperature=0.2,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        raise PlanModelError("Failed to request plan from OpenAI API") from exc

    # The Responses API exposes the combined text via the `output_text` helper.
    raw_output = getattr(response, "output_text", None)
    if not raw_output:
        raise PlanModelError("OpenAI API did not return any content for the plan request")
    return raw_output


def _parse_plan(raw_output: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse the JSON content returned by the language model."""

    try:
        parsed: Dict[str, List[Dict[str, str]]] = json.loads(raw_output)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive parsing
        raise PlanModelError("Received invalid JSON from OpenAI API") from exc

    if "plan" not in parsed or not isinstance(parsed["plan"], list):
        raise PlanModelError("The OpenAI response must contain a 'plan' list")

    return parsed


def _fallback_plan(user_query: str) -> Dict[str, List[Dict[str, str]]]:
    """Construct a very naive plan as a fallback when the API is unavailable."""

    logger.warning("Falling back to heuristic plan generation. API results may differ.")

    # Simple heuristic: outline intent extraction keywords.
    steps = [
        PlanStep("요청 이해: '" + user_query + "' 분석"),
        PlanStep("관련 시스템 또는 페이지 확인"),
        PlanStep("필요한 정보를 수집 또는 실행"),
    ]

    return {"plan": [step.to_dict() for step in steps]}


def get_plan(user_query: str) -> Dict[str, List[Dict[str, str]]]:
    """Generate a structured plan for the provided natural language command."""

    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must be a non-empty string")

    load_dotenv()  # Ensures environment variables from .env are loaded once the function is called.
    api_key = os.getenv("OPENAI_API_KEY")

    try:
        client = _load_client(api_key)
        raw_output = _call_openai_for_plan(client, user_query)
        return _parse_plan(raw_output)
    except PlanModelError:
        logger.exception("Unable to generate plan using OpenAI. Falling back to heuristic output.")
        return _fallback_plan(user_query)


def _format_plan(plan: Dict[str, List[Dict[str, str]]]) -> str:
    """Return a pretty JSON representation of the plan."""

    return json.dumps(plan, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_query = "공지사항 3개 가져와"
    plan_result = get_plan(sample_query)
    print("입력:", sample_query)
    print("생성된 계획:\n" + _format_plan(plan_result))
