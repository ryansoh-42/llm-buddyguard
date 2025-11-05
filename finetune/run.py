import os

from pathlib import Path
from typing import Literal

import torch

from dotenv import find_dotenv, load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from data_handling import convert_csv_to_dataset


load_dotenv(find_dotenv())

MODEL_ID: str = "meta-llama/Llama-3.2-1B-Instruct"
SUBJECTS: list[str] = ["bio", "chem", "physics"]

HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN is not None, "Set HF_TOKEN environment variable!"


def create_directories() -> None:
    paths: list[str] = [
        "metrics/bio",
        "metrics/chem",
        "metrics/physics",
        "models/bio",
        "models/chem",
        "models/physics",
        "results"
    ]

    for p in paths:
        p_obj = Path(p)
        p_obj.mkdir(parents=True, exist_ok=True)


def finetune(
        model_id: str,
        subject: Literal["bio", "chem", "physics"]
) -> None:
    file_str = f"data/{subject}.csv"
    file_obj = Path(file_str)

    assert file_obj.is_file(), f"File {subject}.csv does not exist!"

    dataset = convert_csv_to_dataset(file_str)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="auto",
        token=HF_TOKEN
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # Utility functions
    def apply_chat_template(example: dict[str, str]) -> dict[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["final"]}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return {
            "prompt": prompt
        }


    def tokenize_func(example: dict[str, str]):
        tokens = tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=128)
        tokens["labels"] = [
            -100 if token == tokenizer.pad_token_id else token for token in tokens["input_ids"]
        ]

        return tokens

    formatted_dataset = dataset.map(apply_chat_template)
    formatted_dataset = formatted_dataset.train_test_split(0.05)

    tokenized_dataset = formatted_dataset.map(tokenize_func)
    tokenized_dataset = tokenized_dataset.remove_columns(
        [
            "user",
            "final",
            "prompt"
        ]
    )

    model.train()
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=40,
        logging_steps=40,
        save_steps=10000,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        fp16=False,
        report_to="none",
        log_level="info",
        learning_rate=1e-5,
        max_grad_norm=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer
    )

    trainer.train()

    # trainer.save_model(f"models/{subject}")
    # trainer.save_metrics(f"metrics/{subject}")

    # tokenizer.save_pretrained(f"models/{subject}")

    model.push_to_hub(f"Fawl/is469_project_{subject}", token=HF_TOKEN)
    tokenizer.push_to_hub(f"Fawl/is469_project_{subject}", token=HF_TOKEN)


def main() -> None:
    # Create directories
    create_directories()

    # Run finetuning
    for subject in SUBJECTS:
        finetune(
            MODEL_ID,
            subject=subject
        )


if __name__ == '__main__':
    main()