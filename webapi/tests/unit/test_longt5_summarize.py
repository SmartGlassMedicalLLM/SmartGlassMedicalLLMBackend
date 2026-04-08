"""
Tests for the LongT5 summarization module. The actual model and tokenizer
are replaced by the lightweight stubs registered in conftest.py.
"""

import pytest
from inference.long_t5_summarize import (
    check_token_length,
    summarize,
    MaxTokenLengthExceededException,
    MAX_INPUT_TOKENS,
    PROMPT_PREFIX,
)

# check_token_length

class TestCheckTokenLength:
    def test_short_input_does_not_raise(self):
        """A short input well under the token limit should return a count > 0."""
        count = check_token_length("Aspirin is used for pain.")
        assert isinstance(count, int)
        assert count > 0

    def test_returns_token_count(self):
        """Return value is a positive integer representing the token count."""
        count = check_token_length("hello world")
        assert count > 0

    def test_exceeding_limit_raises(self):
        """Input that exceeds MAX_INPUT_TOKENS should raise MaxTokenLengthExceededException."""
        # Each char ~= 0.25 tokens in the stub, so 4 * MAX + prefix will exceed the limit
        giant_text = "a" * (MAX_INPUT_TOKENS * 4 + len(PROMPT_PREFIX) * 4 + 100)
        with pytest.raises(MaxTokenLengthExceededException) as exc_info:
            check_token_length(giant_text)
        assert "exceed" in str(exc_info.value).lower()

    def test_exception_message_contains_token_count(self):
        """Exception message should mention token counts."""
        giant_text = "a" * (MAX_INPUT_TOKENS * 4 + 1000)
        with pytest.raises(MaxTokenLengthExceededException) as exc_info:
            check_token_length(giant_text)
        msg = str(exc_info.value)
        assert "tokens" in msg

# summarize

class TestSummarize:
    def test_returns_string(self):
        """summarize() should always return a string."""
        result = summarize("This is a short test document about aspirin therapy.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_max_words_zero_no_error(self):
        """max_words=0 means no limit and must not raise."""
        result = summarize("Short text.", max_words=0)
        assert isinstance(result, str)

    def test_max_words_positive_no_error(self):
        """Providing a positive max_words must not raise for reasonable inputs."""
        result = summarize("Short medical text about dosage.", max_words=100)
        assert isinstance(result, str)

    def test_max_words_too_low_raises_value_error(self):
        """max_words so small it produces fewer than 40 tokens must raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            summarize("Some text.", max_words=1)
        assert "too low" in str(exc_info.value).lower()

    def test_oversized_input_raises_max_token_exception(self):
        """Input exceeding the token limit must propagate MaxTokenLengthExceededException."""
        giant_text = "a" * (MAX_INPUT_TOKENS * 4 + 1000)
        with pytest.raises(MaxTokenLengthExceededException):
            summarize(giant_text)

    def test_max_token_exception_is_value_error_subclass(self):
        """MaxTokenLengthExceededException must be a subclass of ValueError."""
        assert issubclass(MaxTokenLengthExceededException, ValueError)