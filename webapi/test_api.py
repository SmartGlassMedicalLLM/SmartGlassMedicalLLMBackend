import json
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from req_res_structures import BaseRequest, BaseResponse, ErrorResponse

app = FastAPI()

class BasicPrompt(BaseModel):
    prompt: str

class ConvoPrompt(BasicPrompt):
    new: bool

class SummarizeInput(BasicPrompt):
    max_words: int

@app.post("/base", response_model=BaseResponse | ErrorResponse)
async def prompt_base(
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
    req = BaseRequest(
        reqRefId=reqRefId,
        resRefId=resRefId,
        prompt=prompt,
        sessionId=sessionId,
        userId=userId,
        docName=docName,
        currPage=currPage,
        highlights=json.loads(highlights) if highlights is not None else None
    )
    if pdf is None:
        return BaseResponse(
            reqRefId = reqRefId,
            resRefId = resRefId,
            answer = "test"
        )
    else:
        return BaseResponse(
            reqRefId = reqRefId,
            resRefId = resRefId,
            answer = "PDF not supported yet"
        )

@app.post("/in-context")
async def prompt_in_context(input: BaseRequest):
    return "test"

@app.post("/fine-tuned")
async def prompt_fine_tuned(input: BaseRequest):
    return {}

@app.post("/convo")
async def prompt_convo_base(input: ConvoPrompt):
    if input.new:
        return {"success": True}
    return {"response": "test"}

@app.post("/summarize")
async def prompt_summarize(input: SummarizeInput):
    result = "test"
    return {"summary": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
