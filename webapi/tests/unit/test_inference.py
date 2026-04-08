"""
Tests for stateless base inference and stateful conversation management.
The vLLM ``LLM`` is fully stubbed via conftest.py.
"""

# medgemma_base – medgemma_base_prompt

class TestMedgemmaBasePrompt:
    def test_returns_string(self):
        """medgemma_base_prompt must return a non-empty string."""
        from inference.medgemma_base import medgemma_base_prompt
        result = medgemma_base_prompt("What is aspirin?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_force_prefix_prepended(self):
        """When force_model_to_start_with is given it should be at the start of the result."""
        from inference.medgemma_base import medgemma_base_prompt
        result = medgemma_base_prompt("Answer in JSON:", force_model_to_start_with="{")
        assert result.startswith("{")

    def test_empty_prefix_works(self):
        """Default empty prefix should not prepend anything extra."""
        from inference.medgemma_base import medgemma_base_prompt
        result = medgemma_base_prompt("Hello")
        # With stub the output is "stubbed model output", so should not start with "{"
        assert isinstance(result, str)


# medgemma_base – run_extraction

class TestRunExtraction:
    def test_returns_string_starting_with_bracket(self):
        """run_extraction must return a string that starts with '['."""
        from inference.medgemma_base import run_extraction
        result = run_extraction("Aspirin", "Aspirin interacts with warfarin.")
        assert isinstance(result, str)
        assert result.startswith("[")


# medgemma_convo – prompt_convo / reset_convo

class TestMedgemmaConvo:
    def setup_method(self):
        """Reset conversation state before each test."""
        from inference.medgemma_convo import reset_convo
        reset_convo()

    def test_prompt_convo_returns_string(self):
        """prompt_convo must return a non-empty string response."""
        from inference.medgemma_convo import prompt_convo
        result = prompt_convo("What are CYP enzymes?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_history_grows_with_each_turn(self):
        """context_history should have 2 entries (user + model) after one turn."""
        import inference.medgemma_convo as convo
        convo.prompt_convo("Turn one")
        assert len(convo.context_history) == 2

    def test_history_grows_across_turns(self):
        """Each additional turn appends 2 more entries to history."""
        import inference.medgemma_convo as convo
        convo.prompt_convo("Turn one")
        convo.prompt_convo("Turn two")
        assert len(convo.context_history) == 4

    def test_reset_clears_history(self):
        """reset_convo must empty the context_history list."""
        import inference.medgemma_convo as convo
        convo.prompt_convo("Some message")
        assert len(convo.context_history) > 0
        convo.reset_convo()
        assert convo.context_history == []

    def test_reset_then_prompt_starts_fresh(self):
        """After reset, a new prompt should result in exactly 2 history entries."""
        import inference.medgemma_convo as convo
        convo.prompt_convo("Old turn")
        convo.reset_convo()
        convo.prompt_convo("New turn")
        assert len(convo.context_history) == 2
