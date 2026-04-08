"""
Singleton model and sampling-parameter configuration for MedGemma

The ``LLM`` object is instantiated **once** at import time so that all
inference modules share a single in-memory model.
"""

from vllm import LLM, SamplingParams

model_id = "google/medgemma-4b-it"

llm = LLM(
    model=model_id,
    dtype="bfloat16",
    max_model_len=8192,
)

base_params    = SamplingParams(temperature=0.01, max_tokens=1024)
extract_params = SamplingParams(temperature=0.0,  max_tokens=2048)
convo_params   = SamplingParams(temperature=0.01, max_tokens=512)