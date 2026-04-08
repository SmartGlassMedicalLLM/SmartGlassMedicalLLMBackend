"""
Shared pytest fixtures and model stubs.

All inference modules import ``vllm`` and ``transformers`` at module level,
which would require GPU hardware in CI. This conftest stubs those packages
out before any test module is imported so the rest of the test suite can run
on any machine.
"""

import sys
import types
import pytest
from unittest.mock import MagicMock

# Stub heavy ML packages before any app module is imported

def _make_vllm_stub():
    """Return a minimal vllm stub that satisfies all import-time references."""
    vllm = types.ModuleType("vllm")

    class _SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Output:
        def __init__(self, text="stubbed model output"):
            self.text = text

    class _RequestOutput:
        def __init__(self, text="stubbed model output"):
            self.outputs = [_Output(text)]

    class _LLM:
        def __init__(self, **kwargs):
            pass
        def generate(self, prompts, params):
            return [_RequestOutput() for _ in prompts]

    vllm.LLM = _LLM
    vllm.SamplingParams = _SamplingParams
    return vllm

def _make_transformers_stub():
    """Return a minimal transformers stub."""
    transformers = types.ModuleType("transformers")

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            inst = cls()
            return inst
        def encode(self, text, **kw):
            # ~1 token per 4 chars — good enough for length checks
            return list(range(max(1, len(text) // 4)))
        def decode(self, ids, **kw):
            return "Summary of the provided text."
        def __call__(self, text, **kw):
            mock = MagicMock()
            mock.__getitem__ = lambda s, k: MagicMock()
            mock.to = lambda *a, **kw: mock
            return mock

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            inst = cls()
            inst.device = "cpu"
            return inst
        def to(self, device):
            return self
        def generate(self, *a, **kw):
            return [[0, 1, 2]]

    transformers.AutoTokenizer = _FakeTokenizer
    transformers.LongT5ForConditionalGeneration = _FakeModel
    return transformers

def _make_torch_stub():
    torch = types.ModuleType("torch")
    torch.cuda = MagicMock()
    torch.cuda.is_available = lambda: False
    return torch

# Register stubs before any app imports happen
for _name, _factory in [
    ("vllm", _make_vllm_stub),
    ("transformers", _make_transformers_stub),
    ("torch", _make_torch_stub),
]:
    if _name not in sys.modules:
        sys.modules[_name] = _factory()

@pytest.fixture
def base_req_payload():
    """Minimal valid BaseRequest JSON payload."""
    return {
        "reqRefId": "req-001",
        "resRefId": "res-001",
        "prompt": "What is metformin?",
    }

@pytest.fixture
def app_client():
    """Return a FastAPI TestClient wired to the real app (models stubbed)."""
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)