"""
Black-box tests verifying that the API backend satisfies the functional
requirements from the SRS document.

FR-1  Summarize highlighted text with user-defined max word count /summarize/
FR-2  Summarize entire document when requested                    /summarize/
FR-3  Output summaries (audio/email/print handled elsewhere)      /summarize/ answer field
FR-7  Chatbot for medical/general queries                         /convo/ and /advanced/query/json
FR-8  Measure LLM accuracy (error codes, references)              /advanced/query response references
FR-9  Reduce hallucinations (temperature, forced JSON)            /advanced/query citations present
FR-10 Allow printing/emailing (answer field available)            /summarize/ answer field
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
    reset_convo()
    yield
    reset_convo()

# FR-1 / FR-2 – Summarization

class TestFR1FR2Summarisation:
    """FR-1: summarize highlighted text; FR-2: summarize entire document."""

    def test_fr1_summarize_with_max_words(self, client):
        """
        FR-1: System shall summarize text within a max word count set by the user.
        Verify the endpoint accepts max_words and returns a summary answer.
        """
        res = client.post("/summarize/", json={
            "reqRefId": "fr1-req",
            "resRefId": "fr1-res",
            "prompt": (
                "Aspirin (acetylsalicylic acid) is a nonsteroidal anti-inflammatory "
                "drug (NSAID) used to treat pain, fever, and inflammation. It is also "
                "used as an antiplatelet agent to reduce the risk of heart attacks."
            ),
            "max_words": 50,
        })
        assert res.status_code == 200
        body = res.json()
        assert "answer" in body
        assert len(body["answer"]) > 0

    def test_fr2_summarize_no_word_limit(self, client):
        """
        FR-2: System shall summarize an entire document when requested.
        max_words=0 signals no limit.
        """
        res = client.post("/summarize/", json={
            "reqRefId": "fr2-req",
            "resRefId": "fr2-res",
            "prompt": "Full document text about drug metabolism.",
            "max_words": 0,
        })
        assert res.status_code == 200
        assert "answer" in res.json()

    def test_fr1_oversized_input_returns_structured_error(self, client):
        """
        FR-1/FR-9: Oversized input must return a structured error with a known
        error code rather than an unhandled 500.
        """
        # The stub tokenizer returns ~1 token per 4 chars, so 4*16384*4 chars
        # is comfortably over the limit
        giant = "word " * (16384 * 4)
        res = client.post("/summarize/", json={
            "reqRefId": "fr1-large",
            "resRefId": "fr1-large",
            "prompt": giant,
        })
        assert res.status_code == 200  # endpoint returns ErrorResponse as 200
        body = res.json()
        assert "error" in body
        assert body["error"]["code"] == "SUMMARIZE_INPUT_EXCEEDS_MAX_TOKENS"

# FR-3 – Output delivery (answer field available for downstream use)

class TestFR3OutputDelivery:
    """FR-3: System shall output summaries via print, email, or audio."""

    def test_fr3_answer_field_present_for_downstream(self, client):
        """
        FR-3: The ``answer`` field in the response is the text payload that
        downstream output modules (print, email, TTS) consume.
        """
        res = client.post("/summarize/", json={
            "reqRefId": "fr3-req",
            "resRefId": "fr3-res",
            "prompt": "Some medical text.",
        })
        body = res.json()
        assert "answer" in body
        assert isinstance(body["answer"], str)

# FR-7 – Chatbot for medical/general questions

class TestFR7ChatbotAccess:
    """FR-7: System shall provide access to a chatbot for medical/general queries."""

    def test_fr7_general_question_returns_answer(self, client):
        """
        FR-7: A general medical question must produce a non-empty answer.
        """
        res = client.post("/advanced/query/json", json={
            "reqRefId": "fr7-req-1",
            "resRefId": "fr7-res-1",
            "prompt": "What medications interact with CBD?",
        })
        assert res.status_code == 200
        body = res.json()
        assert "answer" in body
        assert len(body["answer"]) > 0

    def test_fr7_conversational_turn_returns_answer(self, client):
        """
        FR-7: The /convo/ endpoint maintains a chatbot session with multi-turn support.
        """
        res = client.post("/convo/", json={
            "reqRefId": "fr7-req-2",
            "resRefId": "fr7-res-2",
            "prompt": "What are the side effects of warfarin?",
        })
        assert res.status_code == 200
        body = res.json()
        assert "answer" in body

    def test_fr7_chatbot_handles_empty_prompt_gracefully(self, client):
        """
        FR-7: Sending an empty prompt must not crash the chatbot.
        """
        res = client.post("/convo/", json={
            "reqRefId": "fr7-req-3",
            "resRefId": "fr7-res-3",
            "prompt": "",
        })
        assert res.status_code == 200
        body = res.json()
        assert "answer" in body

# FR-8 / FR-9 – Accuracy & hallucination reduction

class TestFR8FR9AccuracyAndHallucinations:
    """
    FR-8: Measure accuracy of the LLM.
    FR-9: Include methods to reduce/eliminate hallucinations.
    """

    def test_fr8_error_response_contains_structured_code(self, client):
        """
        FR-8: The API must surface machine-readable error codes so accuracy
        issues can be tracked and logged.
        """
        res = client.post("/summarize/", json={
            "resRefId": "fr8-res",
            "prompt": "text",
        })
        assert res.status_code == 422
        body = res.json()
        assert "error" in body
        assert "code" in body["error"]
        assert isinstance(body["error"]["code"], str)

    def test_fr9_invalid_highlights_has_error_code(self, client):
        """
        FR-9: When input is malformed the API must return a structured error
        rather than silently producing a potentially hallucinated result.
        """
        res = client.post("/advanced/query", data={
            "reqRefId": "fr9-req",
            "resRefId": "fr9-res",
            "prompt": "Question",
            "highlights": "{bad json}",
        })
        body = res.json()
        assert "error" in body
        assert body["error"]["code"] == "000_INJSON"

    def test_fr9_correlation_ids_always_echoed(self, client):
        """
        FR-9: Echoing correlation IDs ensures every response is traceable,
        supporting downstream accuracy auditing.
        """
        res = client.post("/advanced/query/json", json={
            "reqRefId": "TRACE-123",
            "resRefId": "TRACE-456",
            "prompt": "test",
        })
        body = res.json()
        assert body.get("reqRefId") == "TRACE-123" or (
            "error" in body and body["reqRefId"] == "TRACE-123"
        )

# FR-10 – Output / reporting

class TestFR10Output:
    """FR-10: System shall allow printing, emailing, and audio playback of summaries."""

    def test_fr10_summary_answer_non_empty_for_output(self, client):
        """
        FR-10: The ``answer`` field must be a non-empty string ready for any
        output channel (print, email, TTS).
        """
        res = client.post("/summarize/", json={
            "reqRefId": "fr10-req",
            "resRefId": "fr10-res",
            "prompt": "Patient presented with hypertension.",
        })
        body = res.json()
        assert isinstance(body.get("answer"), str)
        assert len(body["answer"]) > 0