import torch
import os
from vllm import LLM, SamplingParams

model_id = "google/medgemma-4b-it"

llm = None
base_params    = SamplingParams(temperature=0.01, max_tokens=1024)
extract_params = SamplingParams(temperature=0.0,  max_tokens=2048)
convo_params   = SamplingParams(temperature=0.01, max_tokens=512)

def load_llm():
    global llm
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=8192,
    )

def unload_llm():
    global llm
    del llm
    llm = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()

def initLlm():
    if llm is None:
        load_llm()

def getLlm() -> LLM:
    initLlm()
    return llm

# Load the model on startup only if the env variable "SG_LOAD_MODEL_STARTUP=1"
if os.getenv('SG_LOAD_MODEL_STARTUP', 0) == "1":
    load_llm()