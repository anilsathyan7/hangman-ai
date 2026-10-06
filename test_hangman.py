import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import CanineTokenizer

from config import CANINE_MODEL_NAME
from config import CHAR_TO_ID
from config import LETTERS
from config import MASK_ID
from models import HangmanCANINE
from models import HangmanSlimBERT
from word_index import HangmanWordIndex
from word_index import load_index_words


SLIMBERT_MODEL_DIR = "checkpoints/slimbert"
CANINE_MODEL_DIR = "checkpoints/canine"


class HangmanPlayer:
    """Shared Hangman play loop for Hangman models."""

    def __init__(
        self,
        model_dir: str,
        letter_weight: float = 0.0,
        mlm_weight: float = 1.0,
        use_index: bool = False,
        index_late_fails: int = 2,
        max_index_candidates: int = 200,
    ) -> None:
        """Store common play settings and optional lookup index.

        Args:
            model_dir: Directory containing trainer checkpoints.
            letter_weight: Weight for the 26-way letter-head score.
            mlm_weight: Weight for the position-wise MLM score.
            use_index: Whether to enable late-game word lookup.
            index_late_fails: Remaining fail slots where lookup can start.
            max_index_candidates: Skip lookup when candidates exceed this count.
        """
        self.model_dir = Path(model_dir)
        self.letter_weight = letter_weight
        self.mlm_weight = mlm_weight
        self.use_index = use_index
        self.index_late_fails = index_late_fails
        self.max_index_candidates = max_index_candidates
        self.word_index = HangmanWordIndex(load_index_words()) if use_index else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def best_checkpoint(self) -> Path:
        """Return the best checkpoint path from the latest trainer state.

        Returns:
            Path to the checkpoint marked as best by Hugging Face Trainer.
        """
        checkpoints = sorted(
            self.model_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        state = json.loads((checkpoints[-1] / "trainer_state.json").read_text())
        best_checkpoint = Path(state["best_model_checkpoint"])

        if best_checkpoint.exists():
            return best_checkpoint

        return checkpoints[-1]

    def encode_missed_letters(self, missed: set[str]) -> list[int]:
        """Encode wrong guesses as a 26-way multi-hot vector.

        Args:
            missed: Letters already guessed and found absent.

        Returns:
            List with `1` for missed letters and `0` otherwise.
        """
        return [1 if letter in missed else 0 for letter in LETTERS]

    def encode_pattern(self, pattern: str) -> list[int]:
        """Encode a board pattern for saved eval traces."""
        raise NotImplementedError

    def forward_scores(
        self,
        pattern: str,
        missed: set[str],
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """Return board blank positions, MLM probabilities, and letter scores."""
        raise NotImplementedError

    def guess_letter(
        self,
        pattern: str,
        guessed: set[str],
        missed: set[str],
        fails: int,
        max_fails: int,
    ) -> tuple[str, str]:
        """Choose the next letter from model scores or late-game lookup.

        Args:
            pattern: Current board with `_` for hidden letters.
            guessed: Letters already guessed or revealed.
            missed: Wrong guesses already made.
            fails: Current wrong-guess count.
            max_fails: Maximum wrong guesses allowed.

        Returns:
            Chosen letter and source name: `model` or `index`.
        """
        mask_positions, mlm_probs, letter_scores = self.forward_scores(pattern, missed)
        best_letter = None
        best_score = -1
        model_scores = {}

        # Blend global hidden-letter scores with per-blank MLM probabilities.
        for idx, letter in enumerate(LETTERS):
            if letter in guessed:
                continue

            mlm_score = 1 - torch.prod(1 - mlm_probs[:, CHAR_TO_ID[letter]]).item()
            score = (
                self.letter_weight * letter_scores[idx].item()
                + self.mlm_weight * mlm_score
            )
            model_scores[letter] = score

            if score > best_score:
                best_letter = letter
                best_score = score

        index_letter = self.index_guess(
            pattern,
            guessed,
            missed,
            fails,
            max_fails,
            mask_positions,
            mlm_probs,
            model_scores,
        )

        if index_letter is not None:
            return index_letter, "index"

        return best_letter, "model"

    def word_patterns(self, pattern: str) -> list[tuple[int, str]]:
        """Split a title pattern into words with board offsets.

        Args:
            pattern: Full title board pattern.

        Returns:
            List of `(start_index, word_pattern)` pairs.
        """
        word_patterns = []
        start = 0

        for word_pattern in pattern.split():
            start = pattern.find(word_pattern, start)
            word_patterns.append((start, word_pattern))
            start += len(word_pattern)

        return word_patterns

    def index_guess(
        self,
        pattern: str,
        guessed: set[str],
        missed: set[str],
        fails: int,
        max_fails: int,
        mask_positions: list[int],
        mlm_probs: torch.Tensor,
        model_scores: dict[str, float],
    ) -> str | None:
        """Use the word index to rerank late-game candidate completions.

        Args:
            pattern: Current board with `_` for hidden letters.
            guessed: Letters already guessed or revealed.
            missed: Wrong guesses already made.
            fails: Current wrong-guess count.
            max_fails: Maximum wrong guesses allowed.
            mask_positions: Board positions for blank cells.
            mlm_probs: MLM probabilities at blank positions.
            model_scores: Final per-letter model scores from normal guessing.

        Returns:
            A lookup-backed letter, or `None` when lookup should be skipped.
        """
        if not self.use_index or self.word_index is None:
            return None

        if fails < max_fails - self.index_late_fails:
            return None

        mask_probs = {
            position: mlm_probs[idx]
            for idx, position in enumerate(mask_positions)
        }
        best_letters = None
        best_score = float("-inf")

        for start, word_pattern in self.word_patterns(pattern):
            if "_" not in word_pattern:
                continue

            candidates = sorted(self.word_index.candidates(word_pattern, missed))

            if not candidates or len(candidates) > self.max_index_candidates:
                continue

            for candidate in candidates:
                score = 0
                blanks = 0
                candidate_letters = set()

                for idx, pattern_char in enumerate(word_pattern):
                    if pattern_char != "_":
                        continue

                    letter = candidate[idx]

                    if letter in guessed:
                        candidate_letters = set()
                        break

                    score += torch.log(
                        mask_probs[start + idx][CHAR_TO_ID[letter]].clamp_min(1e-12)
                    ).item()
                    blanks += 1
                    candidate_letters.add(letter)

                if not candidate_letters:
                    continue

                score /= blanks

                if score > best_score:
                    best_score = score
                    best_letters = candidate_letters

        if not best_letters:
            return None

        return max(best_letters, key=lambda letter: model_scores.get(letter, 0))

    def reveal_letter(self, secret: str, pattern: str, letter: str) -> str:
        """Reveal a correct guess everywhere it appears.

        Args:
            secret: Full answer title.
            pattern: Current board pattern.
            letter: Guessed letter.

        Returns:
            Updated board pattern.
        """
        chars = list(pattern)

        for idx, char in enumerate(secret):
            if char == letter:
                chars[idx] = letter

        return "".join(chars)

    def play_game(self, secret: str, pattern: str, max_fails: int = 8) -> dict:
        """Run one Hangman game and return a structured result.

        Args:
            secret: Full answer title.
            pattern: Starting board pattern.
            max_fails: Maximum wrong guesses allowed.

        Returns:
            Result dictionary with win status, counts, final board, and sequence.
        """
        input_pattern = pattern
        guessed = {char for char in pattern if char.isalpha()}
        missed = set()
        fails = 0
        guesses = 0
        sequence = []

        for letter in guessed:
            pattern = self.reveal_letter(secret, pattern, letter)

        start_pattern = pattern

        while "_" in pattern and fails < max_fails:
            letter, source = self.guess_letter(pattern, guessed, missed, fails, max_fails)
            guessed.add(letter)
            guesses += 1

            if letter in secret:
                pattern = self.reveal_letter(secret, pattern, letter)
                result = "hit"
            else:
                missed.add(letter)
                fails += 1
                result = "miss"

            sequence.append(
                {
                    "letter": letter,
                    "result": result,
                    "fails": fails,
                    "missed": sorted(missed),
                    "pattern": pattern,
                    "source": source,
                }
            )

        won = "_" not in pattern
        score = 1 - fails / max_fails if won else 0

        return {
            "won": won,
            "fails": fails,
            "guesses": guesses,
            "score": score,
            "input_pattern": input_pattern,
            "start_pattern": start_pattern,
            "final_pattern": pattern,
            "missed": sorted(missed),
            "sequence": sequence,
        }

    def play(self, secret: str, pattern: str, max_fails: int = 8) -> None:
        """Run one printable Hangman game.

        Args:
            secret: Full answer title.
            pattern: Starting board pattern.
            max_fails: Maximum wrong guesses allowed.
        """
        result = self.play_game(secret, pattern, max_fails)

        print("Secret :", secret)
        print("Start  :", result["start_pattern"])

        for step in result["sequence"]:
            guess_result = "hit " if step["result"] == "hit" else "miss"
            print(
                f"{step['letter']} {step['source']} {guess_result} "
                f"fails={step['fails']}/{max_fails}  {step['pattern']}"
            )

        print("Result :", "won" if result["won"] else "lost")


class HangmanSlimBERTPlayer(HangmanPlayer):
    """Play Hangman with a trained SlimBERT Hangman checkpoint."""

    def __init__(
        self,
        model_dir: str = SLIMBERT_MODEL_DIR,
        letter_weight: float = 0.0,
        mlm_weight: float = 1.0,
        use_index: bool = False,
        index_late_fails: int = 2,
        max_index_candidates: int = 200,
    ) -> None:
        """Load the SlimBERT model and optional word index."""
        super().__init__(
            model_dir=model_dir,
            letter_weight=letter_weight,
            mlm_weight=mlm_weight,
            use_index=use_index,
            index_late_fails=index_late_fails,
            max_index_candidates=max_index_candidates,
        )
        self.model = HangmanSlimBERT().to(self.device)
        self.model.load_state_dict(load_file(self.best_checkpoint() / "model.safetensors"))
        self.model.eval()

    def encode_pattern(self, pattern: str) -> list[int]:
        """Encode a board pattern into SlimBERT token ids.

        Args:
            pattern: Current board with `_` for hidden letters.

        Returns:
            Integer ids where `_` is encoded as `[MASK]`.
        """
        input_ids = []

        for char in pattern:
            if char == "_":
                input_ids.append(MASK_ID)
            else:
                input_ids.append(CHAR_TO_ID[char])

        return input_ids

    def forward_scores(
        self,
        pattern: str,
        missed: set[str],
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """Run SlimBERT and return scores for the current board."""
        input_ids = self.encode_pattern(pattern)
        mask_positions = [
            idx for idx, token_id in enumerate(input_ids) if token_id == MASK_ID
        ]
        model_input = torch.tensor([input_ids], device=self.device)
        attention_mask = torch.ones_like(model_input)
        missed_letters = torch.tensor(
            [self.encode_missed_letters(missed)],
            device=self.device,
            dtype=torch.float,
        )

        with torch.no_grad():
            outputs = self.model(
                model_input,
                attention_mask=attention_mask,
                missed_letters=missed_letters,
            )
            letter_scores = torch.sigmoid(outputs["letter_logits"][0])
            mlm_probs = torch.softmax(outputs["logits"][0, mask_positions], dim=-1)

        return mask_positions, mlm_probs, letter_scores


class HangmanCANINEPlayer(HangmanPlayer):
    """Play Hangman with a trained CANINE Hangman checkpoint."""

    def __init__(
        self,
        model_dir: str = CANINE_MODEL_DIR,
        letter_weight: float = 0.0,
        mlm_weight: float = 1.0,
        use_lora: bool = False,
        use_index: bool = False,
        index_late_fails: int = 2,
        max_index_candidates: int = 200,
    ) -> None:
        """Load the CANINE model, tokenizer, and optional word index."""
        super().__init__(
            model_dir=model_dir,
            letter_weight=letter_weight,
            mlm_weight=mlm_weight,
            use_index=use_index,
            index_late_fails=index_late_fails,
            max_index_candidates=max_index_candidates,
        )
        self.tokenizer = CanineTokenizer.from_pretrained(CANINE_MODEL_NAME)
        self.model = HangmanCANINE(use_lora=use_lora).to(self.device)
        self.model.load_state_dict(load_file(self.best_checkpoint() / "model.safetensors"))
        self.model.eval()

    def encode_pattern(self, pattern: str) -> list[int]:
        """Encode a board pattern with the CANINE tokenizer."""
        return self.tokenizer(pattern)["input_ids"]

    def forward_scores(
        self,
        pattern: str,
        missed: set[str],
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """Run CANINE and return scores for the current board."""
        encoded = self.tokenizer(pattern, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        encoded["missed_letters"] = torch.tensor(
            [self.encode_missed_letters(missed)],
            device=self.device,
            dtype=torch.float,
        )

        mask_positions = [idx for idx, char in enumerate(pattern) if char == "_"]
        # Offset by one because CANINE adds a leading special token.
        output_positions = [idx + 1 for idx in mask_positions]

        with torch.no_grad():
            outputs = self.model(**encoded)
            letter_scores = torch.sigmoid(outputs["letter_logits"][0])
            mlm_probs = torch.softmax(outputs["logits"][0, output_positions], dim=-1)

        return mask_positions, mlm_probs, letter_scores


def build_player(args: argparse.Namespace) -> HangmanPlayer:
    """Create the selected player from parsed CLI args."""
    model_dir = args.model_dir

    if args.model == "slimbert":
        return HangmanSlimBERTPlayer(
            model_dir=model_dir or SLIMBERT_MODEL_DIR,
            letter_weight=args.letter_weight,
            mlm_weight=args.mlm_weight,
            use_index=args.use_index,
            index_late_fails=args.index_late_fails,
            max_index_candidates=args.max_index_candidates,
        )

    return HangmanCANINEPlayer(
        model_dir=model_dir or CANINE_MODEL_DIR,
        letter_weight=args.letter_weight,
        mlm_weight=args.mlm_weight,
        use_lora=args.use_lora,
        use_index=args.use_index,
        index_late_fails=args.index_late_fails,
        max_index_candidates=args.max_index_candidates,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["slimbert", "canine"], default="slimbert")
    parser.add_argument("--secret", default="SLUMDOG AND MILLIONAIRE")
    parser.add_argument("--pattern", default="S______ A__ M__________")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--max-fails", type=int, default=8)
    parser.add_argument("--letter-weight", type=float, default=0.0)
    parser.add_argument("--mlm-weight", type=float, default=1.0)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.set_defaults(use_lora=False)
    parser.add_argument("--use-index", action="store_true")
    parser.add_argument("--index-late-fails", type=int, default=2)
    parser.add_argument("--max-index-candidates", type=int, default=200)
    args = parser.parse_args()

    player = build_player(args)
    player.play(args.secret, args.pattern, args.max_fails)
