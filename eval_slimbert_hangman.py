import argparse
import json
from pathlib import Path

from tqdm import tqdm

from config import DATASET_NAME
from data import build_hangman_dataset
from test_hangman import HangmanPlayer
from test_hangman import HangmanSlimBERTPlayer
from test_hangman import SLIMBERT_MODEL_DIR


FAILURE_DIR = Path("eval_failures")
MODEL_NAME = "slimbert"


def weight_label(value: float) -> str:
    """Make a weight value safe for filenames."""
    return str(value).replace(".", "p")


def save_failures(
    player: HangmanPlayer,
    failures: list[dict],
    max_fails: int,
) -> Path:
    """Save failed games for later inspection."""
    FAILURE_DIR.mkdir(exist_ok=True)
    index_label = (
        f"_index_late_{player.index_late_fails}"
        if player.use_index
        else ""
    )
    path = FAILURE_DIR / (
        f"{MODEL_NAME}_letter_{weight_label(player.letter_weight)}"
        f"_mlm_{weight_label(player.mlm_weight)}{index_label}.json"
    )
    payload = {
        "letter_weight": player.letter_weight,
        "mlm_weight": player.mlm_weight,
        "use_index": player.use_index,
        "index_late_fails": player.index_late_fails,
        "max_index_candidates": player.max_index_candidates,
        "max_fails": max_fails,
        "failures": failures,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def evaluate(
    player: HangmanPlayer,
    dataset_name: str = DATASET_NAME,
    max_fails: int = 8,
) -> None:
    """Evaluate full Hangman games on the fixed test split."""
    hangman_splits = build_hangman_dataset(dataset_name)
    test_hangman = hangman_splits["test"]

    wins = 0
    total_fails = 0
    total_guesses = 0
    total_score = 0
    failures = []
    progress_label = f"letter={player.letter_weight}, mlm={player.mlm_weight}"

    if player.use_index:
        progress_label += f", index=last{player.index_late_fails}"

    for idx in tqdm(range(len(test_hangman)), desc=progress_label):
        secret = test_hangman.dataset[idx]["title"]
        pattern = test_hangman[idx]["pattern"]
        result = player.play_game(secret, pattern, max_fails)

        wins += int(result["won"])
        total_fails += result["fails"]
        total_guesses += result["guesses"]
        total_score += result["score"]

        if not result["won"]:
            failures.append(
                {
                    "idx": idx,
                    "secret": secret,
                    "input_pattern": result["input_pattern"],
                    "input_ids": player.encode_pattern(result["input_pattern"]),
                    "final_missed_letters": player.encode_missed_letters(
                        set(result["missed"]),
                    ),
                    "final_pattern": result["final_pattern"],
                    "fails": result["fails"],
                    "guesses": result["guesses"],
                    "sequence": result["sequence"],
                }
            )

    total = len(test_hangman)
    failure_path = save_failures(player, failures, max_fails)

    print("Weights    :", f"letter={player.letter_weight}, mlm={player.mlm_weight}")
    print("Index      :", "on" if player.use_index else "off")
    if player.use_index:
        print("Index late :", player.index_late_fails)
        print("Max cand   :", player.max_index_candidates)
    print("Games      :", total)
    print("Max fails  :", max_fails)
    print("Win rate   :", round(wins / total, 4))
    print("Avg fails  :", round(total_fails / total, 4))
    print("Avg guesses:", round(total_guesses / total, 4))
    print("Avg score  :", round(total_score / total, 4))
    print("Failures   :", len(failures))
    print("Saved      :", failure_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--model-dir", default=SLIMBERT_MODEL_DIR)
    parser.add_argument("--max-fails", type=int, default=8)
    parser.add_argument("--letter-weight", type=float, default=0.0)
    parser.add_argument("--mlm-weight", type=float, default=1.0)
    parser.add_argument("--use-index", action="store_true")
    parser.add_argument("--index-late-fails", type=int, default=2)
    parser.add_argument("--max-index-candidates", type=int, default=200)
    args = parser.parse_args()

    player = HangmanSlimBERTPlayer(
        model_dir=args.model_dir,
        letter_weight=args.letter_weight,
        mlm_weight=args.mlm_weight,
        use_index=args.use_index,
        index_late_fails=args.index_late_fails,
        max_index_candidates=args.max_index_candidates,
    )

    evaluate(player, dataset_name=args.dataset_name, max_fails=args.max_fails)
