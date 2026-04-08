"""
Exposes ``POST /advanced/query/json`` - a JSON-body variant of the advanced
query that accepts a standard :class:`~utils.req_res_structures.BaseRequest`
and returns a MedGemma answer without PDF context.

Use this endpoint when the caller wants to send a plain medical question
without uploading a document.  For document-grounded Q&A with citations,
use ``POST /advanced/query`` (multipart/form-data).
"""

from fastapi import APIRouter
from utils.req_res_structures import ErrorResponse, BaseRequest, BaseResponse

from inference.medgemma_base import medgemma_base_prompt

router = APIRouter(prefix="/advanced", tags=["advanced"])

@router.post("/query/json", response_model=BaseResponse | ErrorResponse)
async def prompt_base_json(input: BaseRequest):
    """
    Run a single stateless prompt through MedGemma and return the answer.

    :param input: Validated :class:`~utils.req_res_structures.BaseRequest` body.
    :returns: :class:`~utils.req_res_structures.BaseResponse` with the model
              answer in ``answer``.
    """
    return BaseResponse(
        reqRefId=input.reqRefId,
        resRefId=input.resRefId,
        answer=medgemma_base_prompt(input.prompt)
    )
