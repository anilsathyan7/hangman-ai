"""Regenerate README training plots from saved Trainer histories."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".cache/matplotlib")

import matplotlib.pyplot as plt


PLOTS_DIR = Path("plots")
SLIMBERT_STATE_PATH = Path("checkpoints/slimbert/checkpoint-89830/trainer_state.json")
CANINE_STATE_PATH = Path("checkpoints/canine/checkpoint-179660/trainer_state.json")


def load_history(path: Path) -> list[dict]:
    """Load the metric history saved by Hugging Face Trainer."""
    return json.loads(path.read_text())["log_history"]


def metric_points(history: list[dict], metric: str) -> tuple[list[float], list[float]]:
    """Return epoch and value arrays for one logged metric."""
    rows = [row for row in history if metric in row and "epoch" in row]
    return [row["epoch"] for row in rows], [row[metric] for row in rows]


def style_axis(axis: plt.Axes, title: str, ylabel: str) -> None:
    """Apply the shared visual style used by both README plots."""
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(color="#e5e7eb")
    axis.legend()


def save_loss_plot(slimbert_history: list[dict], canine_history: list[dict]) -> None:
    """Save side-by-side training and validation loss curves."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for axis, name, history in zip(axes, ["SlimBERT", "CANINE"], [slimbert_history, canine_history]):
        train_epoch, train_loss = metric_points(history, "loss")
        eval_epoch, eval_loss = metric_points(history, "eval_loss")
        axis.plot(train_epoch, train_loss, color="#2563eb", label="train loss")
        axis.plot(eval_epoch, eval_loss, color="#6b7280", label="validation loss")
        style_axis(axis, name, "Loss")

    figure.suptitle("Training and Validation Loss")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "training_loss.svg", format="svg")
    plt.close(figure)


def save_accuracy_plot(slimbert_history: list[dict], canine_history: list[dict]) -> None:
    """Save validation masked-letter and letter-head accuracy curves."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for axis, metric, title in zip(
        axes,
        ["eval_masked_accuracy", "eval_top1_accuracy"],
        ["Masked Letter Accuracy", "Letter Head Top-1 Accuracy"],
    ):
        for name, history, color in [
            ("SlimBERT", slimbert_history, "#1f77b4"),
            ("CANINE", canine_history, "#ff7f0e"),
        ]:
            epoch, values = metric_points(history, metric)
            axis.plot(epoch, values, color=color, label=name)
        style_axis(axis, title, "Accuracy")

    figure.suptitle("Validation Accuracy")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "validation_accuracy.svg", format="svg")
    plt.close(figure)


if __name__ == "__main__":
    PLOTS_DIR.mkdir(exist_ok=True)
    slimbert_history = load_history(SLIMBERT_STATE_PATH)
    canine_history = load_history(CANINE_STATE_PATH)
    save_loss_plot(slimbert_history, canine_history)
    save_accuracy_plot(slimbert_history, canine_history)
