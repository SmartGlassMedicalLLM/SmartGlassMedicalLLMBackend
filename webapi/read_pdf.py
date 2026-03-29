import json
import fitz
from FastAPI import UploadFile
from req_res_structures import Highlight, DocumentReference
from medgemma_base import medgemma_base_prompt

async def extract_pages(pdf: UploadFile) -> dict[int, str]:
    pdf_bytes = await pdf.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return {i + 1: page.get_text() for i, page in enumerate(doc)}

def get_candidate_passages(
    pages: dict[int, str],
    highlights: list[Highlight]
) -> list[dict]:
    candidates = []
    for h in highlights:
        for page_num in h.pages:
            text = pages.get(page_num, "")
            if h.word.lower() in text.lower():
                candidates.append({
                    "word": h.word,
                    "page": page_num,
                    "text": text[:3000]
                })
    return candidates

def extract_reference(candidate: dict, prompt: str) -> DocumentReference | None:
    extraction_prompt = f"""
You are extracting a citation from a medical document.

User's question: {prompt}
Highlighted term: "{candidate['word']}"
Page {candidate['page']} text:
{candidate['text']}

Find the single most relevant sentence from the page text that supports the
highlighted term in the context of the user's question.

Respond ONLY with valid JSON, no explanation:
{{
  "quote": "<exact sentence from the text>",
  "label": "<short label like 'Study design' or 'Safety profile'>",
  "confidence": <float 0.0-1.0>
}}
If no relevant sentence exists, respond with: {{"quote": null}}
"""
    result = medgemma_base_prompt(extraction_prompt)
    try:
        data = json.loads(result)
        if not data.get("quote"):
            return None
        return DocumentReference(
            refId=f"ref-{candidate['page']}-{candidate['word'][:8]}",
            page=candidate['page'],
            label=data["label"],
            quote=data["quote"],
            highlightedWord=candidate['word'],
            confidence=data["confidence"]
        )
    except Exception:
        return None