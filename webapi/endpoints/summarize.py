"""
Exposes ``POST /summarize/`` for abstractive document summarization powered
by LongT5 (``google/long-t5-tglobal-base``).

The endpoint accepts any text payload and an optional ``max_words`` cap.
Token-length validation is handled inside the inference layer; oversized
inputs receive a structured ``ErrorResponse`` rather than a server crash.
"""

from fastapi import APIRouter
from utils.req_res_structures import ErrorResponse, APIError, BaseRequest, BaseResponse

from inference.long_t5_summarize import summarize, MaxTokenLengthExceededException

router = APIRouter(prefix="/summarize", tags=["summarize"])

class SummarizeInput(BaseRequest):
    """
    Request body for the summarization endpoint.

    Extends :class:`~utils.req_res_structures.BaseRequest` with an optional
    word-count limit.

    :param max_words: Soft upper bound on summary length in words.
        ``0`` (default) means no limit.
    """
    max_words: int = 0

@router.post(
    "/",
    response_model=BaseResponse | ErrorResponse,
    summary="Summarize text or a document passage",
    description=(
        "Generate an abstractive summary of the supplied text using LongT5. "
        "Set `max_words` to a positive integer to cap the summary length; "
        "use `0` for no limit. "
        "Returns `SUMMARIZE_INPUT_EXCEEDS_MAX_TOKENS` if the input is too long "
        "for the model's 16,384-token context window."
    ),
)
async def prompt_summarize(input: SummarizeInput):
    """
    Run LongT5 summarization on ``input.prompt``.

    :param input: Validated :class:`SummarizeInput` request body.
    :returns: :class:`~utils.req_res_structures.BaseResponse` with the summary
              in ``answer``, or :class:`~utils.req_res_structures.ErrorResponse`
              with code ``SUMMARIZE_INPUT_EXCEEDS_MAX_TOKENS`` / ``SUMMARIZE_GENERAL``.
    """
    try:
        result = summarize(input.prompt, max_words=input.max_words)
    except MaxTokenLengthExceededException as e:
        return ErrorResponse(
            reqRefId = input.reqRefId,
            resRefId = input.resRefId,
            error = APIError(
                code = "SUMMARIZE_INPUT_EXCEEDS_MAX_TOKENS",
                message = str(e)
            )
        )
    except Exception as e:
        return ErrorResponse(
            reqRefId = input.reqRefId,
            resRefId = input.resRefId,
            error = APIError(
                code = "SUMMARIZE_GENERAL",
                message = str(e)
            )
        )
    
    return BaseResponse(
        reqRefId=input.reqRefId,
        resRefId=input.resRefId,
        answer=result
    )
