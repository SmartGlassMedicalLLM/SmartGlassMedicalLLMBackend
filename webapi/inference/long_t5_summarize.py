"""
Provides abstractive summarization using ``google/long-t5-tglobal-base``.

The model supports up to :data:`MAX_INPUT_TOKENS` input tokens (16,384), which
makes it suitable for long medical documents.
"""

import torch
from transformers import AutoTokenizer, LongT5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
model = LongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-base").to(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_INPUT_TOKENS = 16384
"""Maximum number of tokens the model can accept as input."""

PROMPT_PREFIX = (
    "Provide a concise abstractive summary that captures the key events, "
    "characters, and moral or conclusion of the following text. "
    "Do not copy sentences directly from the text:\n\n"
)
"""Instruction prefix prepended to every summarization request."""

class MaxTokenLengthExceededException(ValueError):
    """Raised when :data:`MAX_INPUT_TOKENS` is exceeded."""
    pass

def check_token_length(text: str) -> int:
    """
    Validate that the input text (plus the instruction prefix) fits within the
    model's context window.

    :param text: The raw input text to be summarized.
    :returns: The actual token count of the prefixed input.
    :raises MaxTokenLengthExceededException: If the token count exceeds
        :data:`MAX_INPUT_TOKENS`. The message includes how many tokens need
        to be removed.
    """
    prompt = PROMPT_PREFIX + text
    token_count = len(tokenizer.encode(prompt))
    if token_count > MAX_INPUT_TOKENS:
        raise MaxTokenLengthExceededException(
            f"Input too large: {token_count} tokens exceed the "
            f"model's maximum of {MAX_INPUT_TOKENS}. "
            f"Please shorten the input by ~{token_count - MAX_INPUT_TOKENS} tokens."
        )
    return token_count

def summarize(text: str, max_words: int = 0) -> str:
    """
    Generate an abstractive summary of the provided text.

    :param text: The document or passage to summarize.
    :param max_words: Soft upper bound on summary length in words.  The model
        is given ``max_words * 1.5`` output tokens as the ceiling.  Pass
        ``0`` (default) for no word-count constraint (capped at 4k tokens).
    :returns: The decoded summary string.
    :raises MaxTokenLengthExceededException: If *text* is too long for the model.
    :raises ValueError: If *max_words* is so low that fewer than the minimum
        40 output tokens would be allowed.
    """
    # Fail if input is too large
    check_token_length(text)

    prompt = PROMPT_PREFIX + text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    ).to(model.device)

    min_tokens = 40
    max_tokens = int(max_words * 1.5) if max_words > 0 else 4000

    if max_tokens < min_tokens:
        raise ValueError(
            f"max_words={max_words} is too low — would produce fewer than "
            f"{min_tokens} tokens. Use 0 for no limit, or provide a higher value."
        )

    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_tokens,
        min_new_tokens=min_tokens,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        early_stopping=True
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
