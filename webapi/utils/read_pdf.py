"""
Helpers for reading PDF uploads, scoping candidate passages, and running
batched LLM inference to produce cited answers with confidence scores.

Typical call chain::

    pages        = await extract_pages(pdf_upload)
    candidates   = get_candidate_passages(pages, highlights, curr_page)
    answer, refs = pdf_inference_with_references(prompt, pages, candidates)
"""

import json
import fitz
from difflib import SequenceMatcher
from fastapi import UploadFile
from utils.req_res_structures import Highlight, DocumentReference
from inference.medgemma_utils import llm, base_params

async def extract_pages(pdf: UploadFile) -> dict[int, str]:
    """
    Read a FastAPI ``UploadFile`` PDF and return a mapping of page number to
    plain text.

    Pages are 1-indexed. PyMuPDF is used for text extraction.

    :param pdf: The uploaded PDF file received from a multipart/form-data request.
    :returns: ``{page_number: page_text}`` for every page in the document.
    """
    pdf_bytes = await pdf.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return {i + 1: page.get_text() for i, page in enumerate(doc)}

def get_candidate_passages(
    pages: dict[int, str],
    highlights: list[Highlight],
    current_page: int | None
) -> list[dict]:
    """
    Select the most relevant page passages to include in the LLM prompt.

    Scoping rules:

    1. **Highlights present** - only pages that contain a highlighted word are
       returned; each entry is paired with the triggering highlight word.
    2. **Current page only** - if no highlights but a ``current_page`` is given,
       only that page is returned.
    3. **All pages** - fallback when neither highlights nor current page are
       available. Risky as it could use a massive amount of tokens.

    Each returned passage is limited to a maximum of 3k characters (way more
    than a typical page) to ensure a passage isn't too large

    :param pages: Full page map from :func:`extract_pages`.
    :param highlights: Highlighted terms from the user's request.
    :param current_page: 1-indexed current page number, or ``None``.
    :returns: List of dicts with keys ``word``, ``page``, and ``text``.
    """
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
    """
    Compute a similarity score between a model-generated claim and a document
    quote using ``difflib.SequenceMatcher``.

    The score is the ratio of matching characters to the combined length of
    both strings, ranging from 0.0 (no overlap) to 1.0 (identical).

    :param claim: The short phrase from the model's answer being cited.
    :param quote: The verbatim excerpt extracted from the document.
    :returns: Similarity ratio in range ``[0.0, 1.0]``.
    """
    return SequenceMatcher(None, claim.lower(), quote.lower()).ratio()

def build_quote_prompt(citation: dict, pages: dict[int, str]) -> str:
    """
    Build an LLM prompt that asks the model to extract the best supporting
    quote for a citation from its source page.

    :param citation: Dict with keys ``"claim"`` (str) and ``"page"`` (int).
    :param pages: Full page map from :func:`extract_pages`.
    :returns: Prompt string instructing the model to respond with JSON
              ``{"quote": "<excerpt>"}``.
    """
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
    """
    Run two-phase batched LLM inference to answer a question from a PDF and
    produce structured document references with confidence scores.

    First, a single LLM call generates a JSON object containing the full answer text
    and a list of citation objects (``claim``, ``page``, ``label``).

    Second, one LLM call per citation asks the model to extract the verbatim excerpt
    that best supports each claim.  All calls are dispatched as a single batch
    for efficiency.

    Confidence scores are computed via :func:`compute_confidence` by comparing
    each claim to its extracted quote.

    :param prompt: The user's natural-language question.
    :param pages: Full page map from :func:`extract_pages`.
    :param candidates: Scoped passages from :func:`get_candidate_passages`.
    :returns: ``(answer_text, references)`` where ``references`` is a list of
              :class:`~utils.req_res_structures.DocumentReference` objects.
    :raises json.JSONDecodeError: If the model returns malformed JSON in Phase 1.
    """
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