"""Play Hangman with the exported SlimBERT ONNX model."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

import onnxruntime as ort

from config import MASK_ID, ONNX_MODEL_PATH
from test_hangman import HangmanPlayer
from test_hangman import HangmanSlimBERTPlayer


def softmax(values: np.ndarray) -> np.ndarray:
    """Compute stable softmax over the last axis.

    Args:
        values: Raw logits.

    Returns:
        Normalized probabilities with the same shape as `values`.
    """
    shifted = values - values.max(axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute sigmoid for letter-head logits.

    Args:
        values: Raw 26-way letter-head logits.

    Returns:
        Per-letter probabilities with the same shape as `values`.
    """
    return 1 / (1 + np.exp(-values))


def get_providers() -> list[str]:
    """Prefer the installed CUDA provider; otherwise run on CPU.

    Returns:
        Ordered ONNX Runtime execution providers.
    """
    if "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


class HangmanSlimBERTOnnxPlayer(HangmanSlimBERTPlayer):
    """Small ONNX version of the SlimBERT next-letter scorer."""

    def __init__(
        self,
        onnx_model: str = ONNX_MODEL_PATH,
        letter_weight: float = 0.0,
        mlm_weight: float = 1.0,
        use_index: bool = False,
        index_late_fails: int = 2,
        max_index_candidates: int = 200,
    ) -> None:
        """Load an ONNX Runtime session with the usual play settings.

        Args:
            onnx_model: Path to the exported SlimBERT ONNX model.
            letter_weight: Weight for the 26-way letter-head score.
            mlm_weight: Weight for the position-wise MLM score.
            use_index: Whether to enable late-game word lookup.
            index_late_fails: Remaining fail slots where lookup can start.
            max_index_candidates: Skip lookup when candidates exceed this count.
        """
        HangmanPlayer.__init__(
            self,
            model_dir=".",
            letter_weight=letter_weight,
            mlm_weight=mlm_weight,
            use_index=use_index,
            index_late_fails=index_late_fails,
            max_index_candidates=max_index_candidates,
        )
        ort.set_default_logger_severity(3)
        self.session = ort.InferenceSession(
            onnx_model,
            providers=get_providers(),
        )
        print(f"ONNX provider: {self.session.get_providers()[0]}")
        self.forward_time = 0.0
        self.forward_calls = 0

    def average_forward_ms(self) -> float:
        """Return average ONNX model-run time in milliseconds.

        Returns:
            Mean `InferenceSession.run` duration, or `0.0` before a run.
        """
        if self.forward_calls == 0:
            return 0.0

        return self.forward_time * 1000 / self.forward_calls

    def forward_scores(
        self,
        pattern: str,
        missed: set[str],
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """Run ONNX SlimBERT and return scores for the current board.

        Args:
            pattern: Current board with `_` for hidden letters.
            missed: Letters already guessed and found absent.

        Returns:
            Blank positions, MLM probabilities, and 26-way letter scores.
        """
        input_ids = self.encode_pattern(pattern)
        mask_positions = [
            idx for idx, token_id in enumerate(input_ids) if token_id == MASK_ID
        ]
        start_time = time.perf_counter()
        mlm_logits, letter_logits = self.session.run(
            None,
            {
                "input_ids": np.array([input_ids], dtype=np.int64),
                "missed_letters": np.array(
                    [self.encode_missed_letters(missed)],
                    dtype=np.float32,
                ),
            },
        )
        self.forward_time += time.perf_counter() - start_time
        self.forward_calls += 1
        mlm_probs = softmax(mlm_logits[0, mask_positions])
        letter_scores = sigmoid(letter_logits[0])
        return (
            mask_positions,
            torch.tensor(mlm_probs, dtype=torch.float),
            torch.tensor(letter_scores, dtype=torch.float),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test exported SlimBERT ONNX.")
    parser.add_argument("--onnx-model", default=ONNX_MODEL_PATH)
    parser.add_argument("--secret", default="GENUINE EXPERIENCE")
    parser.add_argument("--pattern", default="_EN__NE E__E__EN_E")
    parser.add_argument("--max-fails", type=int, default=8)
    parser.add_argument("--letter-weight", type=float, default=0.0)
    parser.add_argument("--mlm-weight", type=float, default=1.0)
    parser.add_argument("--use-index", action="store_true")
    parser.add_argument("--index-late-fails", type=int, default=2)
    parser.add_argument("--max-index-candidates", type=int, default=200)
    args = parser.parse_args()

    if not Path(args.onnx_model).exists():
        raise FileNotFoundError(
            f"{args.onnx_model} not found. Run export_slimbert_onnx.py first."
        )

    onnx_player = HangmanSlimBERTOnnxPlayer(
        onnx_model=args.onnx_model,
        letter_weight=args.letter_weight,
        mlm_weight=args.mlm_weight,
        use_index=args.use_index,
        index_late_fails=args.index_late_fails,
        max_index_candidates=args.max_index_candidates,
    )
    onnx_player.play(args.secret, args.pattern, args.max_fails)
    print(
        f"Avg ONNX forward: {onnx_player.average_forward_ms():.3f} ms "
        f"over {onnx_player.forward_calls} calls"
    )
