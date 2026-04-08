"""
Exposes ``POST /convo/`` for stateful multi-turn conversation with MedGemma.

The ``new`` flag on the request body can be used to reset the conversation
history before sending the next prompt, starting a fresh session.
"""

from fastapi import APIRouter
from utils.req_res_structures import ErrorResponse, BaseRequest, BaseResponse

from inference.medgemma_convo import prompt_convo, reset_convo

router = APIRouter(prefix="/convo", tags=["convo"])

class ConvoPrompt(BaseRequest):
    """
    Request body for the conversational endpoint.

    :param new: If ``True``, the conversation history is cleared before this
        turn is processed, effectively starting a new session.
    """
    new: bool = False

@router.post("/", response_model=BaseResponse | ErrorResponse)
async def prompt_convo_base(input: ConvoPrompt):
    """
    Handle a single conversational turn.

    If ``input.new`` is ``True`` the history is cleared first.  If
    ``input.prompt`` is empty the endpoint acknowledges the reset without
    calling the model.

    :param input: Validated :class:`ConvoPrompt` request body.
    :returns: :class:`~utils.req_res_structures.BaseResponse` whose ``answer``
              contains the model reply (or an informational message if the
              prompt was empty).
    """
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
