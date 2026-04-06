from fastapi import APIRouter
from webapi.utils.req_res_structures import BaseRequest

from webapi.inference.medgemma_in_context import extract_interactions_from_drug

router = APIRouter(prefix="/in-context", tags=["in-context"])

@router.post("/")
async def prompt_in_context(input: BaseRequest):
    return extract_interactions_from_drug(input.prompt)
