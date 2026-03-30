import json
import fitz
from difflib import SequenceMatcher
from fastapi import UploadFile
from req_res_structures import Highlight, DocumentReference
from medgemma_utils import llm, base_params

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


def compute_confidence(claim: str, quote: str) -> float:
    return SequenceMatcher(None, claim.lower(), quote.lower()).ratio()

def build_quote_prompt(citation: dict, pages: dict[int, str]) -> str:
    page_text = pages.get(citation["page"], "")[:3000]
    return (
        f"From this page text:\n{page_text}\n\n"
        f"Find the excerpt (maximum of 3 consecutive sentences) that best supports "
        f"this claim: \"{citation['claim']}\"\n"
        f"Respond ONLY with valid JSON: {{\"quote\": \"<exact excerpt>\"}}"
    )

def pdf_inference_with_references(
    prompt: str,
    pages: dict[int, str],
    candidates: list[dict],
) -> tuple[str, list[DocumentReference]]:

    extraction_prompt = f"""
You are a medical expert answering a question based solely on the provided document pages.

Question: {prompt}

Document:
{"\n".join(f'[Page {c["page"]}] {c["text"]}' for c in candidates)}

Answer the question accurately and concisely using only information from the document. \
For every factual claim in your answer, identify the page it came from and the short \
phrase from your answer that the citation supports.

Respond ONLY with valid JSON:
{{
  "answer": "<full answer>",
  "citations": [
    {{
      "claim": "<short phrase from your answer being cited>",
      "page": <page number>,
      "label": "<short label e.g. 'Study design', 'Safety profile'>"
    }}
  ]
}}
"""

    formatted_extraction = (
        f"<start_of_turn>user\n{extraction_prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n{{"
    )
    extraction_raw = "{" + llm.generate([formatted_extraction], base_params)[0].outputs[0].text
    result = json.loads(extraction_raw)

    answer = result["answer"]
    citations = result.get("citations", [])

    if not citations:
        return answer, []

    # Build all quote prompts and run as a single batch
    formatted_quote_prompts = [
        f"<start_of_turn>user\n{build_quote_prompt(c, pages)}<end_of_turn>\n"
        f"<start_of_turn>model\n{{"
        for c in citations
    ]

    quote_outputs = llm.generate(formatted_quote_prompts, base_params)

    # Parse results and compute confidence
    references = []
    for i, (citation, output) in enumerate(zip(citations, quote_outputs)):
        try:
            quote_result = json.loads("{" + output.outputs[0].text)
            quote = quote_result.get("quote", "")
        except Exception:
            quote = ""

        references.append(DocumentReference(
            refId=f"ref-{i + 1}",
            page=citation["page"],
            label=citation["label"],
            quote=quote,
            highlightedWord=next(
                (c["word"] for c in candidates if c["page"] == citation["page"] and c["word"]),
                ""
            ),
            confidence=compute_confidence(citation["claim"], quote)
        ))

    return answer, references