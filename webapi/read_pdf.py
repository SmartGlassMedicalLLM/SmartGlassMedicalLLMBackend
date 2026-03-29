import json
import fitz
from fastapi import UploadFile
from req_res_structures import Highlight, DocumentReference
from medgemma_base import medgemma_base_prompt

async def extract_pages(pdf: UploadFile) -> dict[int, str]:
    pdf_bytes = await pdf.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return {i + 1: page.get_text() for i, page in enumerate(doc)}

def get_candidate_passages(
    pages: dict[int, str],
    highlights: list[Highlight],
    current_page: int | None
) -> list[dict]:
    if highlights:
        # If there are highlights, scope to only the pages with highlights
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
    else:
        # If no highlights, scope to current page
        # If no current page, scope to all
        target_pages = [current_page] if current_page else list(pages.keys())
        return [
            {"word": None, "page": p, "text": pages[p][:3000]}
            for p in target_pages
        ]

def extract_reference(candidate: dict, prompt: str) -> DocumentReference | None:
    prompt_highlight_term = "" if candidate["word"] is None else f"""
Highlighted term: "{candidate['word']}"
"""
    extraction_prompt = f"""
You are extracting a citation from a medical document.

User's question: {prompt}{prompt_highlight_term}
Page {candidate['page']} text:
{candidate['text']}
"""
    
    extraction_prompt += """
Find the single most relevant sentence from the page text for the user's question.
""" if candidate["word"] is None else """
Find the single most relevant sentence from the page text that supports the
highlighted term in the context of the user's question.
"""
    
    extraction_prompt += """
The sentence should be in the form of a JSON object with the following keys:

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