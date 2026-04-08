"""
Black-box tests against the live FastAPI app (models stubbed).
Tests verify correct HTTP status codes, response shapes, and error handling
for the two primary JSON-body endpoints.
"""

import pytest
from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)

# POST /summarize/

class TestSummarizeEndpoint:
    def test_valid_request_returns_200(self, client):
        """A well-formed request should return HTTP 200."""
        res = client.post("/summarize/", json={
            "reqRefId": "req-1",
            "resRefId": "res-1",
            "prompt": "Aspirin is a common analgesic used for pain and fever.",
        })
        assert res.status_code == 200

    def test_response_contains_answer(self, client):
        """Response body must include a non-empty ``answer`` field."""
        res = client.post("/summarize/", json={
            "reqRefId": "req-2",
            "resRefId": "res-2",
            "prompt": "Long document about pharmacology.",
        })
        body = res.json()
        assert "answer" in body
        assert isinstance(body["answer"], str)
        assert len(body["answer"]) > 0

    def test_correlation_ids_echoed(self, client):
        """reqRefId and resRefId must be echoed back in the response."""
        res = client.post("/summarize/", json={
            "reqRefId": "my-req-id",
            "resRefId": "my-res-id",
            "prompt": "Some text.",
        })
        body = res.json()
        assert body["reqRefId"] == "my-req-id"
        assert body["resRefId"] == "my-res-id"

    def test_max_words_field_accepted(self, client):
        """max_words field should be accepted without error."""
        res = client.post("/summarize/", json={
            "reqRefId": "req-3",
            "resRefId": "res-3",
            "prompt": "Text to summarize.",
            "max_words": 100,
        })
        assert res.status_code == 200

    def test_missing_prompt_returns_422(self, client):
        """Missing required ``prompt`` field must return HTTP 422."""
        res = client.post("/summarize/", json={
            "reqRefId": "req-4",
            "resRefId": "res-4",
        })
        assert res.status_code == 422

    def test_missing_req_ref_id_returns_422(self, client):
        """Missing required ``reqRefId`` must return HTTP 422."""
        res = client.post("/summarize/", json={
            "resRefId": "res-5",
            "prompt": "Text.",
        })
        assert res.status_code == 422

    def test_422_response_has_error_envelope(self, client):
        """The 422 body must follow the ErrorResponse envelope."""
        res = client.post("/summarize/", json={"resRefId": "r", "prompt": "p"})
        body = res.json()
        assert "error" in body
        assert "code" in body["error"]
        assert body["error"]["code"] == "VALIDATION_ERROR"

# POST /advanced/query/json

class TestAdvancedQueryJsonEndpoint:
    def test_valid_request_returns_200(self, client):
        """A well-formed request should return HTTP 200."""
        res = client.post("/advanced/query/json", json={
            "reqRefId": "req-10",
            "resRefId": "res-10",
            "prompt": "What medications interact with CBD?",
        })
        assert res.status_code == 200

    def test_response_contains_answer(self, client):
        """Response body must contain an ``answer`` string."""
        res = client.post("/advanced/query/json", json={
            "reqRefId": "req-11",
            "resRefId": "res-11",
            "prompt": "Explain metformin.",
        })
        body = res.json()
        assert "answer" in body
        assert isinstance(body["answer"], str)

    def test_correlation_ids_echoed(self, client):
        """Both correlation IDs must be echoed."""
        res = client.post("/advanced/query/json", json={
            "reqRefId": "ABC",
            "resRefId": "XYZ",
            "prompt": "Hello",
        })
        body = res.json()
        assert body["reqRefId"] == "ABC"
        assert body["resRefId"] == "XYZ"

    def test_optional_fields_ignored_gracefully(self, client):
        """Extra optional fields (sessionId, userId) should not cause errors."""
        res = client.post("/advanced/query/json", json={
            "reqRefId": "req-12",
            "resRefId": "res-12",
            "prompt": "Question",
            "sessionId": "sess-1",
            "userId": "user-1",
        })
        assert res.status_code == 200

    def test_missing_prompt_returns_422(self, client):
        """Missing prompt should return 422 with ErrorResponse."""
        res = client.post("/advanced/query/json", json={
            "reqRefId": "req-13",
            "resRefId": "res-13",
        })
        assert res.status_code == 422
