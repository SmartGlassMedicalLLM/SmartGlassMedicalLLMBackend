from fastapi import APIRouter
from utils.req_res_structures import ErrorResponse, BaseRequest, BaseResponse

from inference.medgemma_base import medgemma_base_prompt

router = APIRouter(prefix="/advanced", tags=["advanced"])

@router.post("/query/json", response_model=BaseResponse | ErrorResponse)
async def prompt_base_json(input: BaseRequest):
    return BaseResponse(
        reqRefId=input.reqRefId,
        resRefId=input.resRefId,
        answer=medgemma_base_prompt(input.prompt)
    )
