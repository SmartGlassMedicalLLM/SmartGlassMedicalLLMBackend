from fastapi import FastAPI
from pydantic import BaseModel
import requests

from medgemma_base import medgemma_base_prompt
from medgemma_in_context import extract_interactions_from_drug
from medgemma_convo import prompt_convo, reset_convo

app = FastAPI()

class BasicPrompt(BaseModel):
    prompt: str

class ConvoPrompt(BaseModel):
    prompt: str
    new: bool

@app.post("/base")
async def prompt_base(input: BasicPrompt):
    return medgemma_base_prompt(input.prompt)

@app.post("/in-context")
async def prompt_in_context(input: BasicPrompt):
    return extract_interactions_from_drug(input.prompt)

@app.post("/fine-tuned")
async def prompt_fine_tuned(input: BasicPrompt):
    return {}

@app.post("/convo")
async def prompt_convo_base(input: ConvoPrompt):
    if input.new:
        reset_convo()
        return {"success": True}
    return prompt_convo(input.prompt)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)