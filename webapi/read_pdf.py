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
    print("Candidate: ", candidate, "Prompt: ", prompt)
    prompt_highlight_term = "" if candidate["word"] is None else f"""
Highlighted term: "{candidate['word']}"
"""
    prompt_sentence_extraction = """
Find the single most relevant sentence from the page text for the user's question.
""" if candidate["word"] is None else """
Find the single most relevant sentence from the page text that supports the
highlighted term in the context of the user's question.
"""
    extraction_prompt = f"""
You are extracting a citation from a medical document.

User's question: {prompt}{prompt_highlight_term}
Page {candidate['page']} text:
{candidate['text']}
{prompt_sentence_extraction}
The sentence should be in the form of a JSON object with the following keys:

Respond ONLY with valid JSON, no explanation:
{{
  "quote": "<exact sentence from the text>",
  "label": "<short label like 'Study design' or 'Safety profile'>",
  "confidence": <float 0.0-1.0>
}}
If no relevant sentence exists, respond with: {{"quote": null}}
"""
    result = medgemma_base_prompt(extraction_prompt, "{")
    print("Result: ", result)
    try:
        data = json.loads(result)
        if not data.get("quote"):
            return None
        return DocumentReference(
            refId=f"ref-{candidate['page']}-{candidate['word'][:8].replace(" ", "_")}",
            page=candidate['page'],
            label=data["label"],
            quote=data["quote"],
            highlightedWord=candidate['word'],
            confidence=data["confidence"]
        )
    except Exception:
        return None

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
                (c.word for c in (candidates or []) if citation["page"] in c.pages),
                ""
            ),
            confidence=quote_result["confidence"]
        ))

    return answer, references