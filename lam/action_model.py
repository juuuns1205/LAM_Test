"""Utility module to convert plan steps into executable Selenium actions."""
from __future__ import annotations

import copy
import json
import os
import logging
import difflib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from functools import lru_cache

try:  # pragma: no cover - optional dependency handling
    import torch
except ImportError:  # pragma: no cover - fallback when torch is unavailable
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency handling
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:  # pragma: no cover - fallback when transformers is unavailable
    AutoModelForSeq2SeqLM = AutoTokenizer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency handling
    from jsonformer import Jsonformer
except ImportError:  # pragma: no cover - fallback when jsonformer is unavailable
    Jsonformer = None  # type: ignore[assignment]

if torch is not None:
    import torch.nn as nn

    class _JsonformerSeq2SeqWrapper(nn.Module):
        """Adapter to let encoder-decoder models behave like causal models for Jsonformer."""

        def __init__(self, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer) -> None:
            super().__init__()
            self._model = model
            self._tokenizer = tokenizer
            self.device = next(model.parameters()).device
            config = getattr(model, "config", None)
            decoder_start = getattr(config, "decoder_start_token_id", None)
            if decoder_start is None:
                decoder_start = getattr(config, "bos_token_id", None)
            if decoder_start is None:
                decoder_start = getattr(config, "pad_token_id", None)
            if decoder_start is None:
                decoder_start = getattr(tokenizer, "pad_token_id", None)
            if decoder_start is None:
                decoder_start = getattr(tokenizer, "bos_token_id", None)
            if decoder_start is None:
                decoder_start = 0
            self._decoder_start_token_id = decoder_start

        def forward(self, input_ids):  # type: ignore[override]
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, device=self.device)
            input_ids = input_ids.to(self.device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            decoder_input_ids = torch.full(
                (input_ids.size(0), 1),
                self._decoder_start_token_id,
                dtype=input_ids.dtype,
                device=self.device,
            )
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
            return outputs

        def to(self, *args, **kwargs):  # type: ignore[override]
            self._model.to(*args, **kwargs)
            self.device = next(self._model.parameters()).device
            return self

        def eval(self):  # type: ignore[override]
            self._model.eval()
            return self

        def generate(self, *args, **kwargs):
            return self._model.generate(*args, **kwargs)

else:
    _JsonformerSeq2SeqWrapper = None  # type: ignore[misc]

logger = logging.getLogger(__name__)

_SUPPORTED_ACTIONS = {"click", "input"}
_MODEL_DIR = Path(__file__).resolve().parent.parent / "trained_action_model"
_MODEL_CACHE: Tuple[AutoTokenizer, AutoModelForSeq2SeqLM, Any] | None = None
_MAX_GENERATION_TOKENS = 256
_TRAINING_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "action_dataset.jsonl"


_ACTION_OBJECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(_SUPPORTED_ACTIONS)},
        "selector": {
            "type": "object",
            "properties": {
                "css": {"type": "string"},
                "xpath": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "value": {"type": "string"},
        "timestamp": {"type": "string"},
    },
    "required": ["action", "selector"],
    "additionalProperties": False,
}



def _normalise_step_text(value: str) -> str:
    return " ".join(value.strip().split()).lower()


@lru_cache(maxsize=1)
def _load_training_action_map() -> tuple[dict[str, List[Dict[str, Any]]], List[str]]:
    mapping: dict[str, List[Dict[str, Any]]] = {}
    raw_steps: List[str] = []
    try:
        with _TRAINING_DATA_PATH.open("r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = record.get("step")
                actions = record.get("actions")
                if not isinstance(step, str) or not isinstance(actions, list):
                    continue
                filtered_actions = [action for action in actions if isinstance(action, dict)]
                if not filtered_actions:
                    continue
                mapping[_normalise_step_text(step)] = filtered_actions
                raw_steps.append(step)
    except FileNotFoundError:
        return {}, []
    return mapping, raw_steps


def _lookup_training_actions(step_text: str) -> List[Dict[str, Any]] | None:
    mapping, raw_steps = _load_training_action_map()
    if not mapping:
        return None

    normalised = _normalise_step_text(step_text)
    if normalised in mapping:
        return copy.deepcopy(mapping[normalised])

    matches = difflib.get_close_matches(step_text, raw_steps, n=1, cutoff=0.6)
    if matches:
        matched_step = matches[0]
        return copy.deepcopy(mapping.get(_normalise_step_text(matched_step), [])) or None

    return None


_JSONFORMER_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": copy.deepcopy(_ACTION_OBJECT_SCHEMA),
    "minItems": 1,
}

_TOP_LEVEL_JSONFORMER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "result": copy.deepcopy(_JSONFORMER_SCHEMA),
    },
    "required": ["result"],
    "additionalProperties": False,
}


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

    training_actions = _lookup_training_actions(step_text)
    if training_actions:
        return [_normalise_action(action) for action in training_actions]

    try:
        model_output = run_local_model(step_text)
        if model_output is None or (
            isinstance(model_output, (list, dict)) and not model_output
        ):
            raise ValueError("Model did not return any output")

        parsed_output = model_output
    except Exception as exc:  # pragma: no cover - defensive error handling
        logger.warning("Falling back to heuristic action for step %r due to model error: %s", step_text, exc)
        return _fallback_actions(step_text)

    if isinstance(parsed_output, dict):
        parsed_actions = [parsed_output]
    elif isinstance(parsed_output, list):
        parsed_actions = [action for action in parsed_output if isinstance(action, dict)]
    else:
        logger.warning(
            "Model output for step %r is not a dict or list; falling back to heuristic action", step_text
        )
        return _fallback_actions(step_text)

    normalised_actions = [_normalise_action(action) for action in parsed_actions]
    normalised_actions = [action for action in normalised_actions if action is not None]

    if not normalised_actions:
        logger.warning("Model did not produce valid actions for step %r; using fallback", step_text)
        return _fallback_actions(step_text)

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

    def _clean_optional(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() in {"null", "none"}:
                return None
            return stripped
        return str(value)

    if isinstance(selector_data, dict):
        css_selector = _clean_optional(selector_data.get("css"))
        xpath_selector = _clean_optional(selector_data.get("xpath"))

    value = _clean_optional(action.get("value"))
    timestamp = _clean_optional(action.get("timestamp"))

    return {
        "action": action_type,
        "selector": {"css": css_selector, "xpath": xpath_selector},
        "value": value,
        "timestamp": timestamp,
    }


def _fallback_actions(step_text: str) -> List[Dict[str, Any]]:
    """Return training actions when available, otherwise a simple click fallback."""

    training_actions = _lookup_training_actions(step_text)
    if training_actions:
        return training_actions

    return [
        {
            "action": "click",
            "selector": {"css": f"BUTTON[title='{step_text}']", "xpath": None},
            "value": None,
            "timestamp": None,
        }
    ]


def run_local_model(prompt: str) -> Dict[str, Any] | List[Dict[str, Any]]:
    """Run the fine-tuned model stored on disk to predict actions for a step."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")

    if torch is None:  # pragma: no cover - defensive check
        raise RuntimeError("PyTorch is required to execute the local action model")


    tokenizer, model, device = _load_or_initialise_model()

    if Jsonformer is not None and _JsonformerSeq2SeqWrapper is not None:
        try:
            inner_schema = copy.deepcopy(_JSONFORMER_SCHEMA)
            schema = copy.deepcopy(_TOP_LEVEL_JSONFORMER_SCHEMA)
            schema["properties"]["result"] = inner_schema
            jsonformer_model = _JsonformerSeq2SeqWrapper(model, tokenizer)
            generator = Jsonformer(
                model=jsonformer_model,
                tokenizer=tokenizer,
                json_schema=schema,
                prompt=prompt,
            )

            result = generator()
            if result is None:
                raise ValueError("Model did not return any output")

            return _extract_jsonformer_result(result)
        except Exception as exc:  # pragma: no cover - defensive safety net
            logger.warning("Jsonformer generation failed for prompt %r: %s", prompt, exc)

    return _generate_with_plain_model(prompt, tokenizer, model, device)


def _extract_jsonformer_result(result: Any) -> Dict[str, Any] | List[Dict[str, Any]]:
    """Normalise the Jsonformer output into a list or dictionary of actions."""

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive safety net
            raise ValueError("Model returned a string that is not valid JSON") from exc

    if isinstance(result, dict) and "result" in result:
        result = result["result"]

    if not isinstance(result, (dict, list)):
        raise ValueError("Model output must be a dictionary or a list of dictionaries")

    return result



def _extract_json_fragment(text: str) -> Any:
    """Attempt to recover a JSON object or array embedded within raw text."""

    text = text.strip()

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue

    raise ValueError("Model returned text that is not valid JSON")

def _generate_with_plain_model(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: Any,
) -> Dict[str, Any] | List[Dict[str, Any]]:
    """Fall back to regular text generation when Jsonformer is unavailable."""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generated_tokens = model.generate(
        **inputs,
        max_new_tokens=_MAX_GENERATION_TOKENS,
        num_beams=1,
        do_sample=False,
    )
    decoded = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    if not decoded.strip():
        raise ValueError("Model did not return any output")

    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError:
        try:
            logger.debug("Model produced non-JSON output: %s", decoded)
            parsed = _extract_json_fragment(decoded)
        except ValueError as exc:  # pragma: no cover - defensive safety net
            logger.warning("Could not parse model output as JSON. Falling back to heuristic action: %s", decoded)
            return _fallback_actions(prompt)

    if not isinstance(parsed, (dict, list)):
        logger.warning("Model output was not a dict or list. Falling back to heuristic action: %s", parsed)
        return _fallback_actions(prompt)

    return parsed



def _load_or_initialise_model() -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM, Any]:
    """Load the fine-tuned seq2seq model once and cache it for reuse."""

    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None or torch is None:
        raise ImportError(
            "transformers and torch must be installed to use the local action model"
        )

    if not _MODEL_DIR.exists():
        raise FileNotFoundError(
            f"The trained model directory '{_MODEL_DIR}' does not exist. Run the training script first."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR, use_fast=False)
    except OSError as exc:
        if os.name == "nt":
            logger.warning("Retrying tokenizer load with fast implementation because the slow tokenizer failed: %s", exc)
            tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR, use_fast=True)
        else:
            raise
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_DIR)
    except RuntimeError as exc:
        if 'size mismatch' in str(exc) and (_MODEL_DIR / 'pytorch_model.bin').exists():
            logger.warning("Retrying model load from PyTorch weights because safetensors load failed: %s", exc)
            model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_DIR, use_safetensors=False)
        else:
            raise
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    _MODEL_CACHE = (tokenizer, model, device)
    return _MODEL_CACHE


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
    print("MODEL_DIR:", _MODEL_DIR)

