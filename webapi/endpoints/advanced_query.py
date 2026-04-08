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
from utils.req_res_structures import ErrorResponse, APIError, BaseResponse, Highlight
import json

from inference.medgemma_base import medgemma_base_prompt
from utils.read_pdf import extract_pages, get_candidate_passages, pdf_inference_with_references

router = APIRouter(prefix="/advanced", tags=["advanced"])

@router.post("/query", response_model=BaseResponse | ErrorResponse)
async def prompt_base_form_data(
    reqRefId: str = Form(...),
    resRefId: str = Form(...),
    prompt: str = Form(...),
    sessionId: str | None = Form(None),
    userId: str | None = Form(None),
    docName: str | None = Form(None),
    currPage: int | None = Form(None),
    highlights: str | None = Form(None),
    pdf: UploadFile | None = File(None)
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
        except TypeError:
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
