import json
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from medgemma_base import medgemma_base_prompt
from medgemma_in_context import extract_interactions_from_drug
from medgemma_convo import prompt_convo, reset_convo
from long_t5_summarize import summarize

from req_res_structures import BaseRequest, BaseResponse, ErrorResponse, APIError, Highlight
from read_pdf import extract_pages, get_candidate_passages, pdf_inference_with_references

app = FastAPI()

class BasicPrompt(BaseModel):
    prompt: str

class ConvoPrompt(BasicPrompt):
    new: bool

class SummarizeInput(BasicPrompt):
    max_words: int

@app.post("/base", response_model=BaseResponse | ErrorResponse)
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
    try:
        json_highlights=json.loads(highlights) if highlights is not None else None
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
            reqRefId = reqRefId,
            resRefId = resRefId,
            answer = medgemma_base_prompt(prompt)
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

@app.post("/base/json", response_model=BaseResponse | ErrorResponse)
async def prompt_base_json(input: BaseRequest):
    return BaseResponse(
        reqRefId = input.reqRefId,
        resRefId = input.resRefId,
        answer = "test"
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
