from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
import requests

app = FastAPI()

class Question(BaseModel):
    prompt: str

@app.post("/local")
async def prompt_local_llm(question: Question):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": question.prompt,
            "stream": False
        }
    )
    
    return response.json()

@app.post("/gemini")
async def prompt_gemini_llm(question: Question):
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=question.prompt
    )

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)