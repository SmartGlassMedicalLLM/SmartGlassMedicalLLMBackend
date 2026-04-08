"""
Unit Tests utils/general_utils.py & utils/read_pdf.py

Tests for PDF helpers that do *not* call the LLM:
  - extract_text_from_pdf (mocked fitz)
  - get_candidate_passages
  - compute_confidence
  - build_quote_prompt
"""

import pytest
from unittest.mock import MagicMock, patch
from utils.req_res_structures import Highlight

# Force fitz to be a Mock so that when it is imported (uninstalled in test environment) there are no errors
import sys
sys.modules['fitz'] = MagicMock()

# extract_text_from_pdf

class TestExtractTextFromPdf:
    def test_returns_concatenated_page_text(self):
        """Text from all pages should be concatenated into one string."""
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page one text. "
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page two text."

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page1, mock_page2]))

        with patch("fitz.open", return_value=mock_doc):
            from utils.general_utils import extract_text_from_pdf
            result = extract_text_from_pdf("/fake/path.pdf")

        assert "Page one text." in result
        assert "Page two text." in result

    def test_closes_document_after_read(self):
        """fitz document must be closed regardless of page count."""
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([]))

        with patch("fitz.open", return_value=mock_doc):
            from utils.general_utils import extract_text_from_pdf
            extract_text_from_pdf("/fake/path.pdf")

        mock_doc.close.assert_called_once()

    def test_empty_pdf_returns_empty_string(self):
        """A PDF with no pages should return an empty string."""
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([]))

        with patch("fitz.open", return_value=mock_doc):
            from utils.general_utils import extract_text_from_pdf
            result = extract_text_from_pdf("/fake/empty.pdf")

        assert result == ""

# get_candidate_passages

class TestGetCandidatePassages:
    """Tests for utils.read_pdf.get_candidate_passages."""

    @pytest.fixture(autouse=True)
    def import_fn(self):
        from utils.read_pdf import get_candidate_passages
        self.get_candidate_passages = get_candidate_passages

    def _pages(self):
        return {
            1: "Aspirin reduces fever and pain.",
            2: "CBD interacts with CYP enzymes.",
            3: "Dosage information is listed here.",
        }

    def test_highlights_scope_to_matching_pages(self):
        """When highlights are present only pages containing that word are returned."""
        highlights = [Highlight(word="CBD", pages=[2])]
        result = self.get_candidate_passages(self._pages(), highlights, current_page=None)
        assert len(result) == 1
        assert result[0]["page"] == 2
        assert result[0]["word"] == "CBD"

    def test_highlights_word_not_found_excluded(self):
        """A highlight that doesn't match any text on the specified page is skipped."""
        highlights = [Highlight(word="morphine", pages=[1])]
        result = self.get_candidate_passages(self._pages(), highlights, current_page=None)
        assert result == []

    def test_no_highlights_uses_current_page(self):
        """Without highlights, only the current page is returned."""
        result = self.get_candidate_passages(self._pages(), [], current_page=2)
        assert len(result) == 1
        assert result[0]["page"] == 2
        assert result[0]["word"] is None

    def test_no_highlights_no_current_page_returns_all(self):
        """Without highlights or current page, all pages are returned."""
        result = self.get_candidate_passages(self._pages(), [], current_page=None)
        assert len(result) == 3

    def test_text_truncated_to_3000_chars(self):
        """Each passage text is capped at 3k characters."""
        long_pages = {1: "x" * 5000}
        result = self.get_candidate_passages(long_pages, [], current_page=1)
        assert len(result[0]["text"]) == 3000

    def test_multiple_highlights_multiple_pages(self):
        """Multiple Highlight objects each contribute their own candidates."""
        highlights = [
            Highlight(word="Aspirin", pages=[1]),
            Highlight(word="CBD", pages=[2]),
        ]
        result = self.get_candidate_passages(self._pages(), highlights, current_page=None)
        pages_returned = {r["page"] for r in result}
        assert pages_returned == {1, 2}

# compute_confidence

class TestComputeConfidence:
    @pytest.fixture(autouse=True)
    def import_fn(self):
        from utils.read_pdf import compute_confidence
        self.compute_confidence = compute_confidence

    def test_identical_strings_score_one(self):
        """Identical strings should return a confidence of 1.0."""
        score = self.compute_confidence("aspirin reduces pain", "aspirin reduces pain")
        assert score == pytest.approx(1.0)

    def test_completely_different_strings_score_near_zero(self):
        """Completely unrelated strings should score below 0.2."""
        score = self.compute_confidence("aaaa", "zzzz")
        assert score < 0.2

    def test_partial_overlap_between_zero_and_one(self):
        """Strings with partial overlap should score strictly between 0 and 1."""
        score = self.compute_confidence("aspirin reduces pain", "aspirin is a drug")
        assert 0.0 < score < 1.0

    def test_case_insensitive(self):
        """Comparison must be case-insensitive."""
        score_lower = self.compute_confidence("aspirin", "aspirin")
        score_mixed = self.compute_confidence("ASPIRIN", "aspirin")
        assert score_lower == pytest.approx(score_mixed)

    def test_returns_float(self):
        """Return type is always float."""
        result = self.compute_confidence("hello", "world")
        assert isinstance(result, float)


# build_quote_prompt

class TestBuildQuotePrompt:
    @pytest.fixture(autouse=True)
    def import_fn(self):
        from utils.read_pdf import build_quote_prompt
        self.build_quote_prompt = build_quote_prompt

    def test_prompt_contains_claim(self):
        """The generated prompt should embed the claim text."""
        citation = {"claim": "Aspirin reduces fever", "page": 1}
        pages = {1: "Aspirin reduces fever and pain in adults."}
        prompt = self.build_quote_prompt(citation, pages)
        assert "Aspirin reduces fever" in prompt

    def test_prompt_contains_page_text(self):
        """The generated prompt should include the source page text."""
        citation = {"claim": "some claim", "page": 2}
        pages = {2: "This is the page text."}
        prompt = self.build_quote_prompt(citation, pages)
        assert "This is the page text." in prompt

    def test_prompt_instructs_json_response(self):
        """The prompt should ask the model to respond with JSON."""
        citation = {"claim": "claim", "page": 1}
        pages = {1: "text"}
        prompt = self.build_quote_prompt(citation, pages)
        assert "JSON" in prompt
        assert '"quote"' in prompt

    def test_missing_page_uses_empty_text(self):
        """If the page is not in the map the prompt is still generated (empty text)."""
        citation = {"claim": "claim", "page": 99}
        pages = {1: "other text"}
        prompt = self.build_quote_prompt(citation, pages)
        assert "claim" in prompt  # should not raise
