from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests

from medgemma_base import medgemma_base_prompt
from medgemma_in_context import extract_interactions_from_drug
from medgemma_convo import prompt_convo, reset_convo
from long_t5_summarize import summarize

from req_res_structures import BaseRequest, BaseResponse, ErrorResponse

app = FastAPI()

class BasicPrompt(BaseModel):
    prompt: str

class ConvoPrompt(BasicPrompt):
    new: bool

class SummarizeInput(BasicPrompt):
    max_words: int

@app.post("/base")
async def prompt_base(input: BaseRequest, pdf: UploadFile | None = File(None)):
    if input.pdf is None:
        return BaseResponse(
            reqRefId = input.reqRefId,
            resRefId = input.resRefId,
            answer = medgemma_base_prompt(input.prompt),
        )
    else:
        return BaseResponse(
            reqRefId = input.reqRefId,
            resRefId = input.resRefId,
            answer = "PDF not supported yet",
        )

@app.post("/in-context")
async def prompt_in_context(input: BaseRequest):
    return extract_interactions_from_drug(input.prompt)

@app.post("/fine-tuned")
async def prompt_fine_tuned(input: BaseRequest):
    return {}

@app.post("/convo")
async def prompt_convo_base(input: ConvoPrompt):
    if input.new:
        reset_convo()
        return {"success": True}
    return {"response": prompt_convo(input.prompt)}

@app.post("/summarize")
async def prompt_summarize(input: SummarizeInput):
    result = summarize(input.prompt, max_words=input.max_words)
    return {"summary": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
