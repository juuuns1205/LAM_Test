"""Fine-tune a small seq2seq model to generate Selenium actions from plan steps."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import math

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

DATASET_PATH = Path("data/action_dataset.jsonl")
MODEL_NAME = "t5-small"
OUTPUT_DIR = Path("trained_action_model")
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 256


def _ensure_dataset_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file '{path}' was not found. Please create it before training."
        )


def _convert_actions_to_json(actions: Any) -> str:
    if isinstance(actions, str):
        return actions
    return json.dumps(actions, ensure_ascii=False)


def _preprocess_examples(examples: dict[str, List[Any]], tokenizer: AutoTokenizer) -> dict[str, Any]:
    steps = [step if isinstance(step, str) else "" for step in examples["step"]]
    targets = [_convert_actions_to_json(item) for item in examples["actions"]]

    model_inputs = tokenizer(steps, max_length=MAX_INPUT_LENGTH, truncation=True)
    labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main() -> None:
    _ensure_dataset_exists(DATASET_PATH)
    set_seed(42)

    raw_datasets = load_dataset("json", data_files={"train": str(DATASET_PATH)})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_dataset = raw_datasets["train"].map(
        lambda batch: _preprocess_examples(batch, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    train_example_count = len(tokenized_dataset)
    if train_example_count == 0:
        raise ValueError("The training dataset is empty; add at least one example.")

    per_device_batch_size = max(1, min(4, train_example_count))
    num_train_epochs = 5 if train_example_count < 20 else 30
    steps_per_epoch = max(1, math.ceil(train_example_count / per_device_batch_size))
    logging_steps = max(1, steps_per_epoch)

    print(f"Training on {train_example_count} examples for {num_train_epochs} epoch(s) with batch size {per_device_batch_size}.")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=5e-4,
        weight_decay=0.0,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
