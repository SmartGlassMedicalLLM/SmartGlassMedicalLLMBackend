from fastapi import FastAPI
from pydantic import BaseModel
import requests

from medgemma_base import medgemma_base_prompt
from medgemma_in_context import extract_interactions_from_drug
from long_t5_script import summarize

app = FastAPI()

class Input(BaseModel):
    prompt: str

class SummarizeInput(Input):
    max_words: int

@app.post("/base")
async def prompt_base(input: Input):
    return medgemma_base_prompt(input.prompt)

@app.post("/in-context")
async def prompt_in_context(input: Input):
    return extract_interactions_from_drug(input.prompt)

@app.post("/fine-tuned")
async def prompt_fine_tuned(input: Input):
    return {}

@app.post("/summarize")
async def prompt_summarize(input: SummarizeInput):
    result = summarize(input.prompt, max_words=input.max_words)
    return {"summary": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
