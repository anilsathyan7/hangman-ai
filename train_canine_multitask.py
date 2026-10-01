import argparse

import numpy as np
import torch
from transformers import CanineTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from config import CANINE_MODEL_NAME
from config import DATASET_NAME
from config import IGNORE_INDEX
from data import build_hangman_dataset
from models import HangmanCANINE


tokenizer = CanineTokenizer.from_pretrained(CANINE_MODEL_NAME)


def hangman_canine_multitask_collator(batch: list[dict]) -> dict:
    """Tokenize patterns and keep MLM, letter, and missed-letter tensors.

    Args:
        batch: List of examples with text patterns, MLM labels, and letter labels.

    Returns:
        Dictionary with tokenizer outputs, missed-letter inputs, and labels.
    """
    encoded = tokenizer(
        [example["pattern"] for example in batch],
        padding=True,
        return_tensors="pt",
    )
    max_len = encoded["input_ids"].shape[1]

    labels = []
    for example in batch:
        # CANINE adds special tokens, which are ignored by the MLM loss.
        sample_labels = [IGNORE_INDEX] + example["labels"] + [IGNORE_INDEX]
        pad_len = max_len - len(sample_labels)
        labels.append(sample_labels + [IGNORE_INDEX] * pad_len)

    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    encoded["letter_labels"] = torch.tensor(
        [example["letter_labels"] for example in batch],
        dtype=torch.float,
    )
    encoded["missed_letters"] = torch.tensor(
        [example["missed_letters"] for example in batch],
        dtype=torch.float,
    )
    return encoded


def compute_metrics(eval_pred: tuple) -> dict:
    """Measure masked-letter and next-guess accuracy.

    Args:
        eval_pred: Tuple of model logits and labels.

    Returns:
        Dictionary with MLM masked accuracy and top-1 hidden-letter accuracy.
    """
    predictions, labels = eval_pred
    mlm_logits, letter_logits = predictions
    mlm_labels, letter_labels = labels

    mlm_predictions = np.argmax(mlm_logits, axis=-1)
    mlm_mask = mlm_labels != IGNORE_INDEX
    masked_accuracy = (mlm_predictions[mlm_mask] == mlm_labels[mlm_mask]).mean()

    letter_predictions = np.argmax(letter_logits, axis=-1)
    valid = letter_labels.sum(axis=-1) > 0

    if not valid.any():
        top1_accuracy = 0.0
    else:
        hits = letter_labels[np.arange(len(letter_labels)), letter_predictions] == 1
        top1_accuracy = hits[valid].mean()

    return {
        "masked_accuracy": masked_accuracy,
        "top1_accuracy": top1_accuracy,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--output-dir", default="./hangman_canine_full_plateau")
    parser.add_argument("--run-name", default="hangman-canine-full-plateau")
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    model = HangmanCANINE(
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    learning_rate = args.learning_rate or (5e-4 if args.use_lora else 3e-5)
    min_learning_rate = learning_rate * 0.05

    print("Dataset:", args.dataset_name)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    print("LoRA:", args.use_lora)
    print("Effective batch:", args.batch_size * args.gradient_accumulation_steps)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="reduce_lr_on_plateau",
        lr_scheduler_kwargs={
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "threshold": 0.001,
            "min_lr": min_learning_rate,
        },
        weight_decay=0.01,
        warmup_steps=0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        logging_steps=100,
        report_to="wandb",
        run_name=args.run_name,
        dataloader_num_workers=8,
        remove_unused_columns=False,
        label_names=["labels", "letter_labels"],
    )

    hangman_splits = build_hangman_dataset(args.dataset_name)
    train_hangman = hangman_splits["train"]
    val_hangman = hangman_splits["val"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hangman,
        eval_dataset=val_hangman,
        data_collator=hangman_canine_multitask_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
