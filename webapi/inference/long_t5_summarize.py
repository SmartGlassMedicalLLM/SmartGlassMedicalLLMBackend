import torch
import os
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from medgemma_utils import unload_llm

MODEL_ID = "google/long-t5-tglobal-base"
MAX_INPUT_TOKENS = 16384
PROMPT_PREFIX = (
    "Provide a concise abstractive summary that captures the key events, "
    "characters, and moral or conclusion of the following text. "
    "Do not copy sentences directly from the text:\n\n"
)

tokenizer = None
model = None

def loadT5_tokenizer_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = LongT5ForConditionalGeneration.from_pretrained(MODEL_ID).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

def unloadT5_tokenizer_model():
    global tokenizer, model
    del tokenizer, model
    tokenizer = None
    model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def initT5():
    if tokenizer is None or model is None:
        loadT5_tokenizer_model()

# Load the model on startup only if the env variable "SG_LOAD_MODEL_STARTUP=1"
if os.getenv('SG_LOAD_MODEL_STARTUP', 0) == 1:
    initT5()

class MaxTokenLengthExceededException(ValueError):
    pass

def check_token_length(text):
    """Raises ValueError if the input exceeds the model's token limit."""
    prompt = PROMPT_PREFIX + text
    token_count = len(tokenizer.encode(prompt))
    if token_count > MAX_INPUT_TOKENS:
        raise MaxTokenLengthExceededException(
            f"Input too large: {token_count} tokens exceed the "
            f"model's maximum of {MAX_INPUT_TOKENS}. "
            f"Please shorten the input by ~{token_count - MAX_INPUT_TOKENS} tokens."
        )
    return token_count

def summarize(text, max_words=0):
    # Unload medgemma for vRAM
    unload_llm()

    # Load model if not loaded
    initT5()

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
