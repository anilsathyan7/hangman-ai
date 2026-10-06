"""Export the trained SlimBERT Hangman model to ONNX."""

import argparse
from pathlib import Path

import onnx
import torch
from onnxconverter_common import float16
from torch.export import Dim
from safetensors.torch import load_file

from config import NUM_LETTER_LABELS
from config import ONNX_MODEL_PATH
from config import SLIMBERT_CHECKPOINT
from models import HangmanSlimBERT


class SlimBERTOnnxWrapper(torch.nn.Module):
    """Return tensor outputs instead of the training dictionary."""

    def __init__(self, model: HangmanSlimBERT) -> None:
        """Store the trained SlimBERT model for ONNX export.

        Args:
            model: Loaded SlimBERT model in evaluation mode.
        """
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        missed_letters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the two inference heads as plain tensors.

        Args:
            input_ids: Encoded board tokens with shape `(batch, sequence)`.
            missed_letters: 26-way wrong-guess vectors.

        Returns:
            Position-wise MLM logits and 26-way letter-head logits.
        """
        attention_mask = torch.ones_like(input_ids)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            missed_letters=missed_letters,
        )
        return outputs["logits"], outputs["letter_logits"]


def export_onnx(checkpoint: str, output_path: str, sequence_length: int) -> None:
    """Export SlimBERT to ONNX for browser/runtime inference.

    Args:
        checkpoint: Directory containing `model.safetensors`.
        output_path: Destination path for the `.onnx` model.
        sequence_length: Maximum dynamic board length to support.
    """
    checkpoint_path = Path(checkpoint)
    model = HangmanSlimBERT()
    model.load_state_dict(load_file(checkpoint_path / "model.safetensors"))
    model.eval()

    wrapper = SlimBERTOnnxWrapper(model).eval()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    input_ids = torch.zeros((1, sequence_length), dtype=torch.long)
    missed_letters = torch.zeros((1, NUM_LETTER_LABELS), dtype=torch.float)
    sequence = Dim("sequence", min=1, max=sequence_length)

    torch.onnx.export(
        wrapper,
        (input_ids, missed_letters),
        output,
        input_names=["input_ids", "missed_letters"],
        output_names=["mlm_logits", "letter_logits"],
        dynamic_shapes={
            "input_ids": {1: sequence},
            "missed_letters": {},
        },
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
    )
    print(f"Exported {checkpoint_path} -> {output}")


def convert_to_fp16(input_path: str, output_path: str) -> None:
    """Convert an FP32 ONNX model to FP16 while preserving float32 I/O.

    Args:
        input_path: Path to the exported FP32 ONNX model.
        output_path: Destination path for the FP16 ONNX model.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fp32_model = onnx.load(input_path)
    fp16_model = float16.convert_float_to_float16(fp32_model, keep_io_types=True)
    output_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*fp16_model.graph.value_info, *fp16_model.graph.output)
    }
    # Keep internal Cast targets consistent with the converted tensor types.
    for node in fp16_model.graph.node:
        if (
            node.op_type == "Cast"
            and output_types.get(node.output[0]) == onnx.TensorProto.FLOAT16
        ):
            for attribute in node.attribute:
                if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                    attribute.i = onnx.TensorProto.FLOAT16
    onnx.checker.check_model(fp16_model)
    onnx.save(fp16_model, output)
    print(f"Converted {input_path} -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SlimBERT Hangman to ONNX.")
    parser.add_argument("--checkpoint", default=SLIMBERT_CHECKPOINT)
    parser.add_argument("--output", default=ONNX_MODEL_PATH)
    parser.add_argument("--sequence-length", type=int, default=80)
    parser.add_argument(
        "--fp16-output",
        default=None,
        help="Optional destination for an FP16 copy of the exported model.",
    )
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.output, args.sequence_length)

    if args.fp16_output:
        convert_to_fp16(args.output, args.fp16_output)
