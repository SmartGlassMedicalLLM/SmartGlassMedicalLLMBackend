import json
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from medgemma_base import medgemma_base_prompt
from medgemma_in_context import extract_interactions_from_drug
from medgemma_convo import prompt_convo, reset_convo
from long_t5_summarize import summarize

from req_res_structures import BaseRequest, BaseResponse, ErrorResponse, APIError, Highlight
from read_pdf import extract_pages, get_candidate_passages, pdf_inference_with_references

app = FastAPI()

class ConvoPrompt(BaseModel):
    new: bool = False

class SummarizeInput(BaseModel):
    max_words: int = 0

def simple_medgemma_response(reqRefId: str, resRefId: str, prompt: str) -> BaseResponse:
    return BaseResponse(
        reqRefId=reqRefId,
        resRefId=resRefId,
        answer=medgemma_base_prompt(prompt)
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    data_response = BaseResponse(
        reqRefId="unknown",
        resRefId="unknown",
        error=APIError(
            code="VALIDATION_ERROR",
            message="\n".join([(":".join(err.loc) + "|" + err.msg) for err in exc.errors()])
        )
    )
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(data_response.model_dump_json()),
    )

@app.post("/advanced/query", response_model=BaseResponse | ErrorResponse)
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
        return simple_medgemma_response(reqRefId, resRefId, prompt)
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

@app.post("/advanced/query/json", response_model=BaseResponse | ErrorResponse)
async def prompt_base_json(input: BaseRequest):
    return simple_medgemma_response(input.reqRefId, input.resRefId, input.prompt)

@app.post("/in-context")
async def prompt_in_context(input: BaseRequest):
    return extract_interactions_from_drug(input.prompt)

@app.post("/fine-tuned")
async def prompt_fine_tuned(input: BaseRequest):
    return {}

@app.post("/convo", response_model=BaseRequest | ErrorResponse)
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

@app.post("/summarize", response_model=BaseRequest | ErrorResponse)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
