import random
import re
import unicodedata
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset

from config import CHAR_TO_ID
from config import DATASET_NAME
from config import ID_TO_CHAR
from config import IGNORE_INDEX
from config import LATE_GAME_SAMPLE_RATE
from config import LETTER_TO_LABEL_ID
from config import LETTERS
from config import MAX_LATE_GAME_HIDDEN_LETTERS
from config import MASK_ID
from config import MAX_MISSES
from config import MAX_TITLE_LENGTH
from config import NUM_PROC
from config import NUM_LETTER_LABELS
from config import RANDOM_SEED
from config import VOCAB_SIZE


def normalize_title(title: str) -> str:
    """Normalize one title to the model character set.

    Args:
        title: Raw movie title.

    Returns:
        Uppercase ASCII title containing only `A-Z` and spaces.
    """
    title = title.upper().strip()
    title = unicodedata.normalize("NFKD", title)
    title = title.encode("ascii", "ignore").decode("ascii")
    title = re.sub(r"[-/:&]+", " ", title)
    title = re.sub(r"[^A-Z ]", "", title)
    return re.sub(r"\s+", " ", title).strip()


def clean_hangman_titles(
    dataset_name: str = DATASET_NAME,
) -> Any:
    """Load and clean movie titles for hangman training.

    Args:
        dataset_name: Hugging Face dataset name or local CSV path.

    Returns:
        A Hugging Face dataset with one deduplicated "title" column.
    """
    if Path(dataset_name).suffix == ".csv":
        dataset = load_dataset("csv", data_files=dataset_name)["train"]
    else:
        dataset = load_dataset(dataset_name)["train"]

        # Local CSV is already English-only; HF source needs this filter.
        dataset = dataset.filter(
            lambda example: example["original_language"] == "en",
            num_proc=NUM_PROC,
        )

    # Keep rows with a usable title and no numbers.
    dataset = dataset.filter(
        lambda example: (
            example["title"] is not None
            and example["title"].strip() != ""
            and not re.search(r"\d", example["title"])
        ),
        num_proc=NUM_PROC,
    )

    # Normalize to the character set used by the model.
    def normalize_batch(batch: dict) -> dict:
        """Normalize a batched title column."""
        return {"title": [normalize_title(title) for title in batch["title"]]}

    dataset = dataset.map(normalize_batch, batched=True, num_proc=NUM_PROC)

    # Filter out titles that are too long or empty after normalization.
    dataset = dataset.filter(
        lambda example: 0 < len(example["title"]) <= MAX_TITLE_LENGTH,
        num_proc=NUM_PROC,
    )

    # Deduplicate titles while preserving order.
    unique_titles = list(dict.fromkeys(dataset["title"]))
    return HFDataset.from_dict({"title": unique_titles})


def create_hangman_state(
    title: str,
    rng: random.Random | None = None,
    late_game: bool = False,
) -> dict:
    """Create one MLM-style hangman state for a title.

    Args:
        title: Normalized movie title.
        rng: Optional random generator for reproducible samples.
        late_game: Whether to hide only a few unique letters.

    Returns:
        Dictionary with input ids, text pattern, MLM labels, and letter features.
    """
    rng = rng or random
    words = title.split()
    unique_letters = set(title.replace(" ", ""))

    if late_game:
        hidden_count = rng.randint(
            1,
            min(MAX_LATE_GAME_HIDDEN_LETTERS, len(unique_letters)),
        )
        hidden = set(rng.sample(sorted(unique_letters), hidden_count))
        revealed = unique_letters - hidden
    else:
        revealed = {rng.choice(word) for word in words}
        remaining = sorted(unique_letters - revealed)

        # Always keep at least one unique letter hidden when possible.
        max_extra = max(0, len(remaining) - 1)
        num_extra = rng.randint(0, max_extra)

        if num_extra > 0:
            revealed.update(rng.sample(remaining, num_extra))

    # Simulate already-missed guesses using letters not in the title.
    wrong_letters = sorted(set(LETTERS) - unique_letters)
    num_misses = rng.randint(0, min(MAX_MISSES, len(wrong_letters)))
    missed = set(rng.sample(wrong_letters, num_misses))

    input_ids = []  # Custom BERT input ids with hidden letters as [MASK].
    pattern = []  # Text board for CANINE, with hidden letters as underscores.
    labels = []  # MLM labels; ignored positions use IGNORE_INDEX.
    letter_labels = [0] * NUM_LETTER_LABELS  # 26-way hidden-letter target.
    missed_letters = [0] * NUM_LETTER_LABELS  # 26-way wrong-guess input.

    for letter in missed:
        missed_letters[LETTER_TO_LABEL_ID[letter]] = 1

    for char in title:
        if char == " ":
            input_ids.append(CHAR_TO_ID[" "])
            pattern.append(" ")
            labels.append(IGNORE_INDEX)
        elif char in revealed:
            input_ids.append(CHAR_TO_ID[char])
            pattern.append(char)
            labels.append(IGNORE_INDEX)
        else:
            input_ids.append(MASK_ID)
            pattern.append("_")
            labels.append(CHAR_TO_ID[char])
            letter_labels[LETTER_TO_LABEL_ID[char]] = 1

    return {
        "input_ids": input_ids,
        "pattern": "".join(pattern),
        "labels": labels,
        "letter_labels": letter_labels,
        "missed_letters": missed_letters,
    }


class HangmanTitleDataset:
    """Samples hangman states from cleaned titles."""

    def __init__(
        self,
        dataset: Any,
        seed: int | None = None,
        late_game_rate: float = 0.0,
    ):
        """Store titles and configure optional late-game oversampling.

        Args:
            dataset: Cleaned title dataset.
            seed: Optional seed for fixed samples.
            late_game_rate: Extra fraction of late-game samples.
        """
        self.dataset = dataset
        self.seed = seed
        self.late_game_count = round(
            len(dataset) * late_game_rate / (1 - late_game_rate)
        )

    def __len__(self) -> int:
        """Return regular titles plus any late-game oversamples."""
        return len(self.dataset) + self.late_game_count

    def __getitem__(self, idx: int) -> dict:
        """Create one Hangman state for a dataset index.

        Args:
            idx: Dataset index.

        Returns:
            Dynamic or fixed Hangman state dictionary.
        """
        late_game = idx >= len(self.dataset)
        title_idx = (idx * 9973) % len(self.dataset) if late_game else idx
        title = self.dataset[title_idx]["title"]
        rng = random.Random(self.seed + idx) if self.seed is not None else None
        return create_hangman_state(title, rng, late_game=late_game)


def build_hangman_dataset(
    dataset_name: str = DATASET_NAME,
) -> dict[str, HangmanTitleDataset]:
    """Build train, validation, and test hangman datasets.

    Args:
        dataset_name: Hugging Face dataset name to load.

    Returns:
        Dictionary with "train", "val", and "test" HangmanTitleDataset objects.
    """
    titles = clean_hangman_titles(dataset_name)
    split = titles.train_test_split(test_size=0.2, seed=RANDOM_SEED)
    temp = split["test"].train_test_split(test_size=0.5, seed=RANDOM_SEED)

    title_splits = {
        "train": split["train"],
        "val": temp["train"],
        "test": temp["test"],
    }
    hangman_splits = {
        "train": HangmanTitleDataset(
            title_splits["train"],
            late_game_rate=LATE_GAME_SAMPLE_RATE,
        ),
        # Keep validation fixed so runs are comparable.
        "val": HangmanTitleDataset(title_splits["val"], seed=RANDOM_SEED),
        # Keep test fixed for final model comparisons.
        "test": HangmanTitleDataset(title_splits["test"], seed=RANDOM_SEED + 1),
    }
    return hangman_splits


def inspect_hangman_sample(
    hangman_splits: dict[str, HangmanTitleDataset],
    idx: int = 0,
) -> None:
    """Print one title, its encoded state, and decoded model input.

    Args:
        hangman_splits: Dictionary of dynamic hangman datasets.
        idx: Example index to inspect from the train split.

    Returns:
        None.
    """
    sample = hangman_splits["train"][idx]
    original_title = hangman_splits["train"].dataset[idx]["title"]

    print("\nOriginal:")
    print(original_title)
    print("\ninput_ids:")
    print(sample["input_ids"])
    print("\nlabels:")
    print(sample["labels"])
    print("\nmissed_letters:")
    print(sample["missed_letters"])
    print("\nModel input:")

    # Show masked letters as underscores for easy inspection.
    for token_id in sample["input_ids"]:
        token = ID_TO_CHAR[token_id]
        print("_" if token == "[MASK]" else token, end="")

    print()


if __name__ == "__main__":
    hangman_splits = build_hangman_dataset()

    total_titles = sum(len(split.dataset) for split in hangman_splits.values())
    print(f"Final unique titles: {total_titles:,}")
    print("Train titles :", len(hangman_splits["train"].dataset))
    print("Train samples:", len(hangman_splits["train"]))
    print("Val          :", len(hangman_splits["val"]))
    print("Test         :", len(hangman_splits["test"]))
    print("Vocab size:", VOCAB_SIZE)
    print(CHAR_TO_ID)

    inspect_hangman_sample(hangman_splits)
