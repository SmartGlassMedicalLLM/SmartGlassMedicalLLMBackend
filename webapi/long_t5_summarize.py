import torch
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from general_utils import extract_text_from_pdf

# ── Load Model ────────────────────────────────────────────────
print("Loading Long-T5...")
tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
model = LongT5ForConditionalGeneration.from_pretrained("google/long-t5-tglobal-base").to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Model loaded!")

# ── Summarize Function ────────────────────────────────────────
def summarize(text, max_words=0):
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=16384
    ).to(model.device)

    max_tokens = int(max_words * 1.5) if max_words > 0 else 4000

    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_tokens,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = input("Enter path to PDF: ").strip()
    text = extract_text_from_pdf(pdf_path)

    max_words = int(input("Max words (0 for no limit): "))
    print("\nSummarizing...\n")

    result = summarize(text, max_words=max_words)
    print("Summary:\n")
    print(result)
