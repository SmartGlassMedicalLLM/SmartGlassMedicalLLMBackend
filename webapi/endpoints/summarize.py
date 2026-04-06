from fastapi import APIRouter
from req_res_structures import ErrorResponse, APIError, BaseRequest, BaseResponse

from long_t5_summarize import summarize

router = APIRouter(prefix="/summarize", tags=["summarize"])

class SummarizeInput(BaseRequest):
    max_words: int = 0

@router.post("/", response_model=BaseResponse | ErrorResponse)
async def prompt_summarize(input: SummarizeInput):
    try:
        result = summarize(input.prompt, max_words=input.max_words)
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
