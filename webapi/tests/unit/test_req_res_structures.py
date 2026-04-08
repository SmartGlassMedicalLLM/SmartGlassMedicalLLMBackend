"""
Covers all Pydantic models: field defaults, validation behavior,
serialization, and nested structure correctness.
"""

import pytest
from pydantic import ValidationError
from utils.req_res_structures import (
    Highlight,
    BaseRequest,
    DocumentReference,
    BaseResponse,
    APIError,
    ErrorResponse,
)

# Highlight

class TestHighlight:
    def test_highlight_valid(self):
        """Highlight accepts a word and a non-empty page list."""
        h = Highlight(word="aspirin", pages=[1, 2, 3])
        assert h.word == "aspirin"
        assert h.pages == [1, 2, 3]

    def test_highlight_single_page(self):
        """Highlight with a single page list is valid."""
        h = Highlight(word="CBD", pages=[5])
        assert h.pages == [5]

    def test_highlight_missing_word_raises(self):
        """Omitting ``word`` should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Highlight(pages=[1])

    def test_highlight_missing_pages_raises(self):
        """Omitting ``pages`` should raise a ValidationError."""
        with pytest.raises(ValidationError):
            Highlight(word="drug")


# BaseRequest

class TestBaseRequest:
    def test_base_request_required_fields(self):
        """BaseRequest requires reqRefId, resRefId, and prompt."""
        req = BaseRequest(reqRefId="r1", resRefId="r2", prompt="hello")
        assert req.reqRefId == "r1"
        assert req.resRefId == "r2"
        assert req.prompt == "hello"

    def test_base_request_optional_fields_default_to_none(self):
        """All optional fields default to None."""
        req = BaseRequest(reqRefId="r1", resRefId="r2", prompt="hello")
        assert req.sessionId is None
        assert req.userId is None
        assert req.docName is None
        assert req.currPage is None
        assert req.highlights is None

    def test_base_request_optional_fields_accepted(self):
        """Optional fields are stored when provided."""
        req = BaseRequest(
            reqRefId="r1",
            resRefId="r2",
            prompt="test",
            sessionId="s1",
            userId="u1",
            docName="doc.pdf",
            currPage=3,
            highlights=[{"word": "CBD", "pages": [1]}],
        )
        assert req.sessionId == "s1"
        assert req.currPage == 3
        assert len(req.highlights) == 1
        assert req.highlights[0].word == "CBD"

    def test_base_request_missing_prompt_raises(self):
        """Missing prompt should raise ValidationError."""
        with pytest.raises(ValidationError):
            BaseRequest(reqRefId="r1", resRefId="r2")


# DocumentReference

class TestDocumentReference:
    def test_document_reference_valid(self):
        """DocumentReference stores all fields correctly."""
        ref = DocumentReference(
            refId="ref-1",
            page=2,
            label="Dosage",
            quote="Take once daily.",
            highlightedWord="dosage",
            confidence=0.85,
        )
        assert ref.refId == "ref-1"
        assert ref.page == 2
        assert ref.confidence == 0.85

    def test_document_reference_confidence_bounds(self):
        """Confidence must be between 0 and 1 (inclusive)."""
        with pytest.raises(ValidationError):
            DocumentReference(
                refId="ref-1", page=1, label="L", quote="q",
                highlightedWord="w", confidence=1.5
            )
        with pytest.raises(ValidationError):
            DocumentReference(
                refId="ref-1", page=1, label="L", quote="q",
                highlightedWord="w", confidence=-0.1
            )

    def test_document_reference_zero_confidence(self):
        """Zero confidence is a valid boundary value."""
        ref = DocumentReference(
            refId="ref-1", page=1, label="L", quote="q",
            highlightedWord="w", confidence=0.0
        )
        assert ref.confidence == 0.0


# BaseResponse

class TestBaseResponse:
    def test_base_response_no_references(self):
        """References defaults to None when not provided."""
        res = BaseResponse(reqRefId="r1", resRefId="r2", answer="42")
        assert res.answer == "42"
        assert res.references is None

    def test_base_response_with_references(self):
        """References list is stored when provided."""
        ref = DocumentReference(
            refId="ref-1", page=1, label="L", quote="q",
            highlightedWord="w", confidence=0.9
        )
        res = BaseResponse(reqRefId="r1", resRefId="r2", answer="ans", references=[ref])
        assert len(res.references) == 1

    def test_base_response_serialises(self):
        """model_dump produces a plain dict suitable for JSON serialisation."""
        res = BaseResponse(reqRefId="r1", resRefId="r2", answer="hello")
        d = res.model_dump()
        assert d["answer"] == "hello"
        assert "references" in d


# APIError & ErrorResponse

class TestErrorModels:
    def test_api_error_fields(self):
        """APIError stores code and message."""
        err = APIError(code="ERR_001", message="Something went wrong")
        assert err.code == "ERR_001"
        assert err.message == "Something went wrong"

    def test_error_response_valid(self):
        """ErrorResponse embeds an APIError correctly."""
        err_res = ErrorResponse(
            reqRefId="r1",
            resRefId="r2",
            error=APIError(code="ERR_001", message="bad input"),
        )
        assert err_res.error.code == "ERR_001"
        assert err_res.reqRefId == "r1"

    def test_error_response_missing_error_raises(self):
        """Omitting the error field raises ValidationError."""
        with pytest.raises(ValidationError):
            ErrorResponse(reqRefId="r1", resRefId="r2")

    def test_error_response_serialises(self):
        """model_dump produces a nested dict with the error sub-object."""
        err_res = ErrorResponse(
            reqRefId="r1", resRefId="r2",
            error=APIError(code="C", message="M")
        )
        d = err_res.model_dump()
        assert d["error"]["code"] == "C"
