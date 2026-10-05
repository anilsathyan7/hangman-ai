"""Run the hard-word Hangman benchmark with both saved models."""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from test_hangman import CANINE_MODEL_DIR
from test_hangman import SLIMBERT_MODEL_DIR
from test_hangman import HangmanCANINEPlayer
from test_hangman import HangmanPlayer
from test_hangman import HangmanSlimBERTPlayer


HARD_TEST_PATH = "datasets/hangman_hard_test.csv"


def starting_patterns(title: str) -> list[str]:
    """Return one starting board for each distinct letter in a title.

    A revealed letter appears in every matching position, as it would in a
    normal Hangman game. For example, ``JAZZ`` yields ``J___``, ``_A__``, and
    ``__ZZ``.

    Args:
        title: Full hard-test answer.

    Returns:
        One board pattern per distinct alphabetic letter in the title.
    """
    letters = sorted({char for char in title if char.isalpha()})

    return [
        "".join(char if char == letter or not char.isalpha() else "_" for char in title)
        for letter in letters
    ]


def evaluate(
    player: HangmanPlayer,
    titles: list[str],
    max_fails: int,
) -> dict[str, float]:
    """Evaluate every one-letter-revealed starting state in the hard set.

    Args:
        player: Loaded SlimBERT or CANINE Hangman player.
        titles: Hard-test answer titles.
        max_fails: Maximum wrong guesses allowed per game.

    Returns:
        Aggregate Hangman metrics for all generated starting states.
    """
    wins = 0
    total_fails = 0
    total_guesses = 0
    total_score = 0.0

    # Treat every distinct initially revealed letter as a separate game state.
    total_games = sum(len(starting_patterns(title)) for title in titles)

    for title in tqdm(titles, desc=player.__class__.__name__):
        for pattern in starting_patterns(title):
            result = player.play_game(title, pattern, max_fails)
            wins += int(result["won"])
            total_fails += result["fails"]
            total_guesses += result["guesses"]
            total_score += result["score"]

    return {
        "games": total_games,
        "win_rate": wins / total_games,
        "avg_fails": total_fails / total_games,
        "avg_guesses": total_guesses / total_games,
        "avg_score": total_score / total_games,
        "failures": total_games - wins,
    }


def print_metrics(model_name: str, metrics: dict[str, float], max_fails: int) -> None:
    """Print hard-benchmark metrics in the normal evaluation format.

    Args:
        model_name: Display name for the evaluated model.
        metrics: Aggregate metrics returned by :func:`evaluate`.
        max_fails: Maximum wrong guesses allowed per game.
    """
    print(f"\n{model_name}")
    print("Games      :", int(metrics["games"]))
    print("Max fails  :", max_fails)
    print("Win rate   :", round(metrics["win_rate"], 4))
    print("Avg fails  :", round(metrics["avg_fails"], 4))
    print("Avg guesses:", round(metrics["avg_guesses"], 4))
    print("Avg score  :", round(metrics["avg_score"], 4))
    print("Failures   :", int(metrics["failures"]))


def result_path(output_dir: str, max_fails: int, use_index: bool) -> Path:
    """Return a distinct CSV path for one hard-benchmark configuration.

    Args:
        output_dir: Directory for saved metric CSV files.
        max_fails: Maximum wrong guesses allowed per game.
        use_index: Whether the word-index fallback is enabled.

    Returns:
        Output path that identifies the fail limit and index setting.
    """
    index_label = "on" if use_index else "off"
    return Path(output_dir) / f"hard_hangman_fails_{max_fails}_index_{index_label}.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-test-path", default=HARD_TEST_PATH)
    parser.add_argument("--slimbert-model-dir", default=SLIMBERT_MODEL_DIR)
    parser.add_argument("--canine-model-dir", default=CANINE_MODEL_DIR)
    parser.add_argument("--max-fails", type=int, default=8)
    parser.add_argument("--letter-weight", type=float, default=0.0)
    parser.add_argument("--mlm-weight", type=float, default=1.0)
    parser.add_argument("--use-index", action="store_true")
    parser.add_argument("--index-late-fails", type=int, default=2)
    parser.add_argument("--max-index-candidates", type=int, default=200)
    parser.add_argument("--output-dir", default="eval_hard_results")
    args = parser.parse_args()

    # The hard set has one title per row; all starting boards are derived below.
    titles = pd.read_csv(args.hard_test_path)["title"].tolist()

    # Both models use the same scoring and optional lookup settings.
    player_options = {
        "letter_weight": args.letter_weight,
        "mlm_weight": args.mlm_weight,
        "use_index": args.use_index,
        "index_late_fails": args.index_late_fails,
        "max_index_candidates": args.max_index_candidates,
    }
    players = {
        "SlimBERT": HangmanSlimBERTPlayer(
            model_dir=args.slimbert_model_dir,
            **player_options,
        ),
        "CANINE": HangmanCANINEPlayer(
            model_dir=args.canine_model_dir,
            **player_options,
        ),
    }

    print("Hard titles:", len(titles))
    print("Index      :", "on" if args.use_index else "off")

    results = []

    for model_name, player in players.items():
        metrics = evaluate(player, titles, args.max_fails)
        print_metrics(model_name, metrics, args.max_fails)
        results.append(
            {
                "model": model_name,
                "hard_titles": len(titles),
                "max_fails": args.max_fails,
                "use_index": args.use_index,
                "index_late_fails": args.index_late_fails,
                **metrics,
            },
        )

    # Keep each fail-limit and index setting in its own result file.
    output_path = result_path(args.output_dir, args.max_fails, args.use_index)
    output_path.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print("\nSaved      :", output_path)
