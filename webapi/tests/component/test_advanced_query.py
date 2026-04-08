"""
Tests for the form-data advanced query endpoint, including highlight parsing
and error handling.
"""

import json
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)

class TestAdvancedQueryMultipart:
    def test_no_pdf_returns_200(self, client):
        """Without a PDF the endpoint should still return 200 with an answer."""
        res = client.post("/advanced/query", data={
            "reqRefId": "req-30",
            "resRefId": "res-30",
            "prompt": "What is CBD?",
        })
        assert res.status_code == 200
        body = res.json()
        assert "answer" in body

    def test_no_pdf_correlation_ids_echoed(self, client):
        """Correlation IDs must be echoed even when no PDF is supplied."""
        res = client.post("/advanced/query", data={
            "reqRefId": "MULTIPART-REQ",
            "resRefId": "MULTIPART-RES",
            "prompt": "Question",
        })
        body = res.json()
        assert body["reqRefId"] == "MULTIPART-REQ"
        assert body["resRefId"] == "MULTIPART-RES"

    def test_invalid_highlights_json_returns_error(self, client):
        """Malformed highlights JSON must return the 000_INJSON error code."""
        res = client.post("/advanced/query", data={
            "reqRefId": "req-31",
            "resRefId": "res-31",
            "prompt": "Question",
            "highlights": "not-valid-json",
        })
        body = res.json()
        assert "error" in body
        assert body["error"]["code"] == "000_INJSON"

    def test_highlights_wrong_schema_returns_error(self, client):
        """highlights JSON with wrong object shape must return 002_INMODE."""
        bad_highlights = json.dumps([{"wrong_key": "value"}])
        res = client.post("/advanced/query", data={
            "reqRefId": "req-32",
            "resRefId": "res-32",
            "prompt": "Question",
            "highlights": bad_highlights,
        })
        body = res.json()
        assert "error" in body
        assert body["error"]["code"] == "002_INMODE"

    def test_valid_highlights_json_accepted(self, client):
        """Valid highlights JSON should not produce an error."""
        valid_highlights = json.dumps([{"word": "aspirin", "pages": [1]}])
        res = client.post("/advanced/query", data={
            "reqRefId": "req-33",
            "resRefId": "res-33",
            "prompt": "Question about aspirin",
            "highlights": valid_highlights,
        })
        # Should not return a highlights-parsing error
        body = res.json()
        if "error" in body:
            assert body["error"]["code"] not in ("000_INJSON", "002_INMODE")

    def test_missing_required_fields_returns_422(self, client):
        """Missing prompt must return 422."""
        res = client.post("/advanced/query", data={
            "reqRefId": "req-34",
            "resRefId": "res-34",
        })
        assert res.status_code == 422
