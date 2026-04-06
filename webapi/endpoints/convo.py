from fastapi import APIRouter
from req_res_structures import ErrorResponse, BaseRequest, BaseResponse

from medgemma_convo import prompt_convo, reset_convo

router = APIRouter(prefix="/convo", tags=["convo"])

class ConvoPrompt(BaseRequest):
    new: bool = False

@router.post("/", response_model=BaseResponse | ErrorResponse)
async def prompt_convo_base(input: ConvoPrompt):
    reset_text = ""
    if input.new:
        reset_convo()
        reset_text = "The conversation was successfully reset. "
    result = None
    if input.prompt != "":
        result = prompt_convo(input.prompt)
    return BaseResponse(
        reqRefId=input.reqRefId,
        resRefId=input.resRefId,
        answer=result if result is not None else f"{reset_text}The prompt is an empty string."
    )
