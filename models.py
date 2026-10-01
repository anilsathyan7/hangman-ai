from torch import nn
from transformers import BertConfig
from transformers import BertForMaskedLM
from transformers import CanineModel

from config import BERT_CONFIG
from config import CANINE_MODEL_NAME
from config import IGNORE_INDEX
from config import NUM_LETTER_LABELS
from config import VOCAB_SIZE


class HangmanSlimBERT(nn.Module):
    """Slim BERT with MLM and 26-letter heads."""

    def __init__(self) -> None:
        """Build the small BERT encoder and both prediction heads."""
        super().__init__()
        mlm_model = BertForMaskedLM(BertConfig(**BERT_CONFIG))
        self.config = mlm_model.config
        self.bert = mlm_model.bert
        self.mlm_head = mlm_model.cls
        # Untie duplicated MLM params so safetensors can save this custom module.
        self.mlm_head.predictions.decoder.weight = nn.Parameter(
            self.mlm_head.predictions.decoder.weight.detach().clone(),
        )
        self.mlm_head.predictions.decoder.bias = nn.Parameter(
            self.mlm_head.predictions.decoder.bias.detach().clone(),
        )
        self.mlm_head.predictions.bias = nn.Parameter(
            self.mlm_head.predictions.bias.detach().clone(),
        )
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.missed_projection = nn.Linear(
            NUM_LETTER_LABELS,
            self.config.hidden_size,
            bias=False,
        )
        self.letter_classifier = nn.Linear(
            self.config.hidden_size,
            NUM_LETTER_LABELS,
        )
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.letter_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        missed_letters=None,
        labels=None,
        letter_labels=None,
    ) -> dict:
        """Run one multitask forward pass.

        Args:
            input_ids: SlimBERT token ids for the current board.
            attention_mask: Mask for real tokens vs padding.
            missed_letters: Optional 26-way wrong-guess input.
            labels: Optional MLM labels with ignored positions as `-100`.
            letter_labels: Optional 26-way hidden-letter targets.

        Returns:
            Dictionary with optional loss, MLM logits, and letter logits.
        """
        if missed_letters is None:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
            missed_embeds = self.missed_projection(missed_letters.float()).unsqueeze(1)
            outputs = self.bert(
                inputs_embeds=inputs_embeds + missed_embeds,
                attention_mask=attention_mask,
            )

        sequence_output = outputs.last_hidden_state
        mlm_logits = self.mlm_head(sequence_output)

        mask = attention_mask.unsqueeze(-1)
        pooled = (sequence_output * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1)
        letter_logits = self.letter_classifier(self.dropout(pooled))

        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fn(
                mlm_logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
            )

        letter_loss = None
        if letter_labels is not None:
            letter_loss = self.letter_loss_fn(letter_logits, letter_labels.float())

        loss = None
        if mlm_loss is not None and letter_loss is not None:
            loss = mlm_loss + letter_loss
        elif mlm_loss is not None:
            loss = mlm_loss
        elif letter_loss is not None:
            loss = letter_loss

        return {
            "loss": loss,
            "logits": mlm_logits,
            "letter_logits": letter_logits,
        }


class HangmanCANINE(nn.Module):
    """CANINE with MLM and 26-letter heads."""

    def __init__(
        self,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        """Load CANINE and attach Hangman MLM and letter heads.

        Args:
            use_lora: Whether to wrap CANINE with LoRA adapters.
            lora_r: LoRA rank.
            lora_alpha: LoRA scaling value.
            lora_dropout: Dropout used inside LoRA adapters.
        """
        super().__init__()
        self.canine = CanineModel.from_pretrained(CANINE_MODEL_NAME)
        self.canine_config = self.canine.config
        if use_lora:
            try:
                from peft import LoraConfig
                from peft import TaskType
                from peft import get_peft_model
            except ImportError as exc:
                raise ImportError("Install peft to train CANINE with LoRA.") from exc

            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "key", "value", "dense"],
            )
            self.canine = get_peft_model(self.canine, lora_config)

        self.dropout = nn.Dropout(self.canine_config.hidden_dropout_prob)
        self.missed_projection = nn.Linear(
            NUM_LETTER_LABELS,
            self.canine_config.hidden_size,
            bias=False,
        )
        self.mlm_classifier = nn.Linear(self.canine_config.hidden_size, VOCAB_SIZE)
        self.letter_classifier = nn.Linear(
            self.canine_config.hidden_size,
            NUM_LETTER_LABELS,
        )
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.letter_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        missed_letters=None,
        labels=None,
        letter_labels=None,
    ) -> dict:
        """Run one CANINE multitask forward pass.

        Args:
            input_ids: CANINE tokenizer ids for the current board.
            attention_mask: Mask for real tokens vs padding.
            token_type_ids: Optional segment ids from the tokenizer.
            missed_letters: Optional 26-way wrong-guess input.
            labels: Optional MLM labels with ignored positions as `-100`.
            letter_labels: Optional 26-way hidden-letter targets.

        Returns:
            Dictionary with optional loss, MLM logits, and letter logits.
        """
        if missed_letters is None:
            outputs = self.canine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            canine_model = (
                self.canine.get_base_model()
                if hasattr(self.canine, "get_base_model")
                else self.canine
            )
            inputs_embeds = canine_model.char_embeddings._embed_hash_buckets(
                input_ids,
                canine_model.config.hidden_size,
                canine_model.config.num_hash_functions,
                canine_model.config.num_hash_buckets,
            )
            missed_embeds = self.missed_projection(missed_letters.float()).unsqueeze(1)
            outputs = self.canine(
                inputs_embeds=inputs_embeds + missed_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        sequence_output = outputs.last_hidden_state
        mlm_logits = self.mlm_classifier(self.dropout(sequence_output))

        mask = attention_mask.unsqueeze(-1)
        pooled = (sequence_output * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1)
        letter_logits = self.letter_classifier(self.dropout(pooled))

        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fn(
                mlm_logits.reshape(-1, VOCAB_SIZE),
                labels.reshape(-1),
            )

        letter_loss = None
        if letter_labels is not None:
            letter_loss = self.letter_loss_fn(letter_logits, letter_labels.float())

        loss = None
        if mlm_loss is not None and letter_loss is not None:
            loss = mlm_loss + letter_loss
        elif mlm_loss is not None:
            loss = mlm_loss
        elif letter_loss is not None:
            loss = letter_loss

        return {
            "loss": loss,
            "logits": mlm_logits,
            "letter_logits": letter_logits,
        }
