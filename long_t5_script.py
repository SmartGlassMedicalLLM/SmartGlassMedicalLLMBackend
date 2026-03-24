import torch
import fitz
from transformers import AutoTokenizer, LongT5ForConditionalGeneration

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

# ── Extract Text from PDF ─────────────────────────────────────
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    return full_text

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    pdf_path = input("Enter path to PDF: ").strip()
    text = extract_text_from_pdf(pdf_path)

    max_words = int(input("Max words (0 for no limit): "))
    print("\nSummarizing...\n")

    result = summarize(text, max_words=max_words)
    print("Summary:\n")
    print(result)
