import argparse

import numpy as np
import torch
from transformers import Trainer
from transformers import TrainingArguments

from config import DATASET_NAME
from config import IGNORE_INDEX
from config import PAD_ID
from data import build_hangman_dataset
from models import HangmanSlimBERT


def hangman_slimbert_collator(batch: list[dict]) -> dict:
    """Pad samples and keep both MLM and letter targets.

    Args:
        batch: List of examples with inputs, MLM labels, and letter labels.

    Returns:
        Dictionary with padded inputs, missed-letter inputs, and labels.
    """
    max_len = max(len(example["input_ids"]) for example in batch)

    input_ids = []
    attention_mask = []
    labels = []
    letter_labels = []
    missed_letters = []

    for example in batch:
        seq_len = len(example["input_ids"])
        pad_len = max_len - seq_len

        input_ids.append(example["input_ids"] + [PAD_ID] * pad_len)
        attention_mask.append([1] * seq_len + [0] * pad_len)
        labels.append(example["labels"] + [IGNORE_INDEX] * pad_len)
        letter_labels.append(example["letter_labels"])
        missed_letters.append(example["missed_letters"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "missed_letters": torch.tensor(missed_letters, dtype=torch.float),
        "labels": torch.tensor(labels, dtype=torch.long),
        "letter_labels": torch.tensor(letter_labels, dtype=torch.float),
    }


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
    parser.add_argument("--output-dir", default="./hangman_slimbert_final")
    parser.add_argument("--run-name", default="hangman-slimbert")
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    model = HangmanSlimBERT()

    print("Dataset:", args.dataset_name)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=1024,
        learning_rate=3e-4,
        lr_scheduler_type="reduce_lr_on_plateau",
        lr_scheduler_kwargs={
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "threshold": 0.001,
            "min_lr": 1.5e-5,
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

    # Dynamic train data; fixed validation boards for comparable runs.
    hangman_splits = build_hangman_dataset(args.dataset_name)
    train_hangman = hangman_splits["train"]
    val_hangman = hangman_splits["val"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hangman,
        eval_dataset=val_hangman,
        data_collator=hangman_slimbert_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
