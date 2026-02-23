from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class Input(BaseModel):
    prompt: str

@app.post("/base")
async def prompt_base(input: Input):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": input.prompt,
            "stream": False
        }
    )
    
    return response.json()

@app.post("/in-context")
async def prompt_in_context(input: Input):
    return {}

@app.post("/fine-tuned")
async def prompt_fine_tuned(input: Input):
    return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)