"""
Tests for the stateful conversational endpoint. Each test class calls
``reset_convo()`` in its setup to ensure a clean state.
"""

import pytest
from fastapi.testclient import TestClient
from inference.medgemma_convo import reset_convo

@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)

@pytest.fixture(autouse=True)
def clean_convo():
    """Reset conversation history before every test."""
    reset_convo()
    yield
    reset_convo()

class TestConvoEndpoint:
    def test_valid_prompt_returns_200(self, client):
        """A well-formed prompt should return HTTP 200."""
        res = client.post("/convo/", json={
            "reqRefId": "req-20",
            "resRefId": "res-20",
            "prompt": "What is ibuprofen?",
        })
        assert res.status_code == 200

    def test_response_contains_answer(self, client):
        """Response must include a non-empty answer string."""
        res = client.post("/convo/", json={
            "reqRefId": "req-21",
            "resRefId": "res-21",
            "prompt": "Describe warfarin.",
        })
        body = res.json()
        assert "answer" in body
        assert isinstance(body["answer"], str)
        assert len(body["answer"]) > 0

    def test_new_flag_resets_and_responds(self, client):
        """Setting new=true should reset history and still return an answer."""
        # First turn to populate history
        client.post("/convo/", json={
            "reqRefId": "r1", "resRefId": "r1", "prompt": "First turn"
        })
        # Reset and ask a new question
        res = client.post("/convo/", json={
            "reqRefId": "req-22",
            "resRefId": "res-22",
            "prompt": "Second session question",
            "new": True,
        })
        assert res.status_code == 200
        body = res.json()
        assert "answer" in body

    def test_new_true_empty_prompt_returns_reset_message(self, client):
        """new=true with empty prompt should confirm reset without model call."""
        res = client.post("/convo/", json={
            "reqRefId": "req-23",
            "resRefId": "res-23",
            "prompt": "",
            "new": True,
        })
        body = res.json()
        assert "reset" in body["answer"].lower()

    def test_empty_prompt_without_reset_returns_info_message(self, client):
        """Empty prompt without new flag should return an informational message."""
        res = client.post("/convo/", json={
            "reqRefId": "req-24",
            "resRefId": "res-24",
            "prompt": "",
        })
        body = res.json()
        assert "empty string" in body["answer"].lower()

    def test_correlation_ids_echoed(self, client):
        """Correlation IDs must be echoed in the response."""
        res = client.post("/convo/", json={
            "reqRefId": "CONV-REQ",
            "resRefId": "CONV-RES",
            "prompt": "Test",
        })
        body = res.json()
        assert body["reqRefId"] == "CONV-REQ"
        assert body["resRefId"] == "CONV-RES"

    def test_missing_prompt_returns_422(self, client):
        """Missing prompt field must return 422."""
        res = client.post("/convo/", json={
            "reqRefId": "req-25",
            "resRefId": "res-25",
        })
        assert res.status_code == 422
