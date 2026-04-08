"""
Exposes ``POST /advanced/query`` - a multipart/form-data endpoint that accepts
an optional PDF upload alongside the question prompt.

* **Without PDF** - delegates directly to MedGemma for a general medical answer.
* **With PDF** - extracts page text, scopes candidate passages to any
  highlighted terms or the current page, then runs two-phase cited inference
  via :func:`~utils.read_pdf.pdf_inference_with_references`.

All fields mirror :class:`~utils.req_res_structures.BaseRequest` but are
received as individual ``Form(...)`` fields (required by FastAPI when mixing
form data with file uploads).
"""

from fastapi import APIRouter, Form, File, UploadFile
from pydantic import ValidationError
from utils.req_res_structures import ErrorResponse, APIError, BaseResponse, Highlight
import json

from inference.medgemma_base import medgemma_base_prompt
from utils.read_pdf import extract_pages, get_candidate_passages, pdf_inference_with_references

router = APIRouter(prefix="/advanced", tags=["advanced"])

@router.post(
    "/query",
    response_model=BaseResponse | ErrorResponse,
    summary="Ask a medical question with optional PDF context",
    description=(
        "Submit a natural-language question with an optional PDF document. "
        "When a PDF is supplied, the answer is grounded in the document and "
        "includes `references` with page numbers, quotes, and confidence scores. "
        "Pass `highlights` as a JSON-encoded array of `{word, pages}` objects to "
        "scope inference to specific highlighted regions. "
        "\n\nError codes:\n"
        "- `000_INJSON` - `highlights` field is not valid JSON.\n"
        "- `002_INMODE` - `highlights` JSON does not match the expected schema.\n"
        "- `001_PDFANS` - inference failed during PDF question-answering."
    ),
)
async def prompt_base_form_data(
    reqRefId: str = Form(..., description="Client-assigned request correlation ID."),
    resRefId: str = Form(..., description="Client-assigned response correlation ID."),
    prompt: str = Form(..., description="The natural-language question to answer."),
    sessionId: str | None = Form(None, description="Optional session identifier."),
    userId: str | None = Form(None, description="Optional user identifier."),
    docName: str | None = Form(None, description="Optional document name for logging."),
    currPage: int | None = Form(None, description="1-indexed current page number."),
    highlights: str | None = Form(None, description="JSON-encoded array of highlight objects."),
    pdf: UploadFile | None = File(None, description="Optional PDF document for grounded Q&A.")
):
    """
    Handle a document-grounded or general medical question.

    Parses the optional ``highlights`` JSON string, then either runs a plain
    MedGemma prompt (no PDF) or PDF-grounded inference with document references.

    :returns: :class:`~utils.req_res_structures.BaseResponse` (with optional
              ``references``) on success, or
              :class:`~utils.req_res_structures.ErrorResponse` on failure.
    """
    try:
        json_highlights = json.loads(highlights) if highlights is not None else None
    except json.JSONDecodeError:
        return ErrorResponse(
            reqRefId = reqRefId,
            resRefId = resRefId,
            error = APIError(
                code = "000_INJSON", # Invalid JSON
                message = "Invalid highlights JSON"
            )
        )

    parsed_highlights = None
    if json_highlights is not None:
        try:
            parsed_highlights = [Highlight(**h) for h in json_highlights]
        except (TypeError, ValidationError):
            return ErrorResponse(
                reqRefId = reqRefId,
                resRefId = resRefId,
                error = APIError(
                    code = "002_INMODE", # Invalid model
                    message = "Invalid internal highlights JSON"
                )
            )

    if pdf is None:
        return BaseResponse(
            reqRefId=reqRefId,
            resRefId=resRefId,
            answer=medgemma_base_prompt(prompt)
        )
    else:
        pages = await extract_pages(pdf)
        candidates = get_candidate_passages(pages, parsed_highlights or [], currPage)
        try:
            answer, references = pdf_inference_with_references(prompt, pages, candidates)
        except Exception as e:
            return ErrorResponse(
                reqRefId = reqRefId,
                resRefId = resRefId,
                error = APIError(
                    code = "001_PDFANS",
                    message = str(e)
                )
            )
        return BaseResponse(
            reqRefId=reqRefId,
            resRefId=resRefId,
            answer=answer,
            references=references
        )
