from pathlib import Path

import pandas as pd

from config import LETTERS


INDEX_PATH = Path("datasets/index.csv")


def load_index_words(path: Path = INDEX_PATH) -> list[str]:
    """Load words from the saved index CSV.

    Args:
        path: CSV file with a `word` column.

    Returns:
        Words in index order.
    """
    index_df = pd.read_csv(path, keep_default_na=False)
    return index_df["word"].tolist()


class HangmanWordIndex:
    """Fast word-candidate lookup for revealed Hangman patterns."""

    def __init__(self, words: list[str]) -> None:
        """Build inverted indexes by length, position, and letter.

        Args:
            words: Clean uppercase words used for lookup.
        """
        self.words_by_length = {}
        self.position_index = {}
        self.letter_index = {}

        for word in words:
            self.words_by_length.setdefault(len(word), set()).add(word)

            for idx, letter in enumerate(word):
                self.position_index.setdefault((len(word), idx, letter), set()).add(word)
                self.letter_index.setdefault((len(word), letter), set()).add(word)

    def candidates(self, pattern: str, missed: set[str]) -> set[str]:
        """Return words matching a single-word pattern and missed letters.

        Args:
            pattern: Single word pattern with `_` for hidden letters.
            missed: Letters already guessed and found absent.

        Returns:
            Candidate words that match the pattern and avoid missed letters.
        """
        candidates = set(self.words_by_length.get(len(pattern), set()))

        for idx, letter in enumerate(pattern):
            if letter != "_":
                candidates &= self.position_index.get((len(pattern), idx, letter), set())

        for letter in missed:
            candidates -= self.letter_index.get((len(pattern), letter), set())

        return candidates

    def score_word(
        self,
        pattern: str,
        guessed: set[str],
        missed: set[str],
    ) -> dict[str, float]:
        """Score next letters from candidates for one word.

        Args:
            pattern: Single word pattern with `_` for hidden letters.
            guessed: Letters already guessed or revealed.
            missed: Letters already guessed and found absent.

        Returns:
            Candidate-count score for each unguessed letter.
        """
        scores = {letter: 0.0 for letter in LETTERS if letter not in guessed}

        for word in self.candidates(pattern, missed):
            for idx, pattern_char in enumerate(pattern):
                letter = word[idx]

                if pattern_char == "_" and letter not in guessed:
                    scores[letter] += 1

        return scores

    def score_pattern(
        self,
        pattern: str,
        guessed: set[str],
        missed: set[str],
    ) -> dict[str, float]:
        """Score next letters across all words in a full title pattern.

        Args:
            pattern: Full title pattern.
            guessed: Letters already guessed or revealed.
            missed: Letters already guessed and found absent.

        Returns:
            Candidate-count score for each unguessed letter.
        """
        scores = {letter: 0.0 for letter in LETTERS if letter not in guessed}

        for word_pattern in pattern.split():
            word_scores = self.score_word(word_pattern, guessed, missed)

            for letter, score in word_scores.items():
                scores[letter] += score

        return scores

    def guess_letter(
        self,
        pattern: str,
        guessed: set[str],
        missed: set[str],
    ) -> str | None:
        """Pick the highest-scoring unguessed letter from the index.

        Args:
            pattern: Full title pattern.
            guessed: Letters already guessed or revealed.
            missed: Letters already guessed and found absent.

        Returns:
            Best index-only guess, or `None` when no candidate contributes.
        """
        scores = self.score_pattern(pattern, guessed, missed)

        if not scores:
            return None

        best_letter = max(scores, key=scores.get)

        if scores[best_letter] == 0:
            return None

        return best_letter


if __name__ == "__main__":
    index_words = load_index_words()
    word_index = HangmanWordIndex(index_words)

    print("Index words:", len(index_words))

    examples = [
        ("GEN_INE EXPERIENCE", {"A", "D", "F", "M", "S", "T", "V"}),
        ("_AR_ELION", set()),
    ]

    for pattern, missed in examples:
        guessed = {letter for letter in pattern if letter.isalpha()} | missed
        scores = word_index.score_pattern(pattern, guessed, missed)

        print()
        print("Pattern    :", pattern)
        print("Missed     :", sorted(missed))
        print("Best guess :", word_index.guess_letter(pattern, guessed, missed))
        print("Top scores :", sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10])
