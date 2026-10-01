import os
import string


# Dataset and preprocessing settings.
DATASET_NAME = "ada-datadruids/full_tmdb_movies_dataset"
CANINE_MODEL_NAME = "google/canine-c"
MAX_TITLE_LENGTH = 80
# Training samples can represent game states before the final allowed miss.
MAX_MISSES = 7
# Extra late-game train samples as a share of the final train mix.
LATE_GAME_SAMPLE_RATE = 0.3
MAX_LATE_GAME_HIDDEN_LETTERS = 3
RANDOM_SEED = 40
NUM_PROC = os.cpu_count()

# Token vocabulary for model inputs and MLM labels.
SPECIAL_TOKENS = ["[PAD]", "[MASK]"]
LETTERS = list(string.ascii_uppercase)
CHARS = LETTERS + [" "]
VOCAB = SPECIAL_TOKENS + CHARS

# Letter-label vocabulary for direct A-Z hangman guesses.
LETTER_TO_LABEL_ID = {letter: idx for idx, letter in enumerate(LETTERS)}
CHAR_TO_ID = {char: idx for idx, char in enumerate(VOCAB)}
ID_TO_CHAR = {idx: char for char, idx in CHAR_TO_ID.items()}

PAD_ID = CHAR_TO_ID["[PAD]"]
MASK_ID = CHAR_TO_ID["[MASK]"]
IGNORE_INDEX = -100
VOCAB_SIZE = len(VOCAB)
NUM_LETTER_LABELS = len(LETTERS)

# Small BERT config shared by the MLM and letter-head models.
BERT_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
    "max_position_embeddings": 128,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "pad_token_id": PAD_ID,
    "type_vocab_size": 1,
}
