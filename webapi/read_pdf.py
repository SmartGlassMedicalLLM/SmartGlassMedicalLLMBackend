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

def pdf_inference_with_references(
    prompt: str,
    pages: dict[int, str],
    candidates: list[dict],
) -> tuple[str, list[DocumentReference]]:
    extraction_prompt = f"""
{prompt}

Your answer should be concise and academic/medical. Do not start with or follow with any conversational text. Be accurate, specific, and informative. Do so while keeping answers per page short and to the point.
Document text (relevant pages):
{"\n".join(f"[Page {c['page']}] {c['text']}" for c in candidates)}

Respond ONLY with valid JSON:
{{
  "answer": "<your full answer here>",
  "citations": [
    {{
      "claim": "<exact phrase from your answer that needs citation>",
      "page": <page number>,
      "label": "<short label>"
    }}
  ]
}}
"""
    result = json.loads(medgemma_base_prompt(extraction_prompt, "{"))
    answer = result["answer"]

    references = []
    for i, citation in enumerate(result.get("citations", [])):
        page_text = pages.get(citation["page"], "")
        quote_prompt = f"""
From this page text:
{page_text[:3000]}

Find the excerpt (maximum of 3 consecutive sentences) that best supports this claim: "{citation['claim']}"
Respond ONLY with valid JSON: {{"quote": "<exact sentence>", "confidence": <0.0-1.0>}}
"""
        quote_result = json.loads(medgemma_base_prompt(quote_prompt, "{"))
        references.append(DocumentReference(
            refId = f"ref-{citation['page']}-{citation['label'][:8].lower().replace(" ", "_")}",
            page = citation["page"],
            label = citation["label"],
            quote = quote_result["quote"],
            highlightedWord=next(
                (c.word for c in (candidates or []) if citation["page"] in pages),
                ""
            ),
            confidence=quote_result["confidence"]
        ))

    return answer, references