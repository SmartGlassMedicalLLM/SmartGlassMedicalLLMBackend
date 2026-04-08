"""
Simple inference using MedGemma.

This module exposes two functions:

* :func:`run_extraction` - drug-interaction extraction returning a JSON list.
* :func:`medgemma_base_prompt` - general-purpose prompt→completion helper.

Both use the shared :data:`~inference.medgemma_utils.llm` singleton so the
model is loaded only once at process startup.
"""

from inference.medgemma_utils import llm, base_params, extract_params

def run_extraction(target_drug: str, target_text: str) -> str:
    """
    Extract drug-drug interactions from prescribing text using a zero-shot
    structured-output prompt.

    The model is forced to begin its response with ``[`` so that the output
    can be parsed directly as a JSON array.

    :param target_drug: Name of the drug whose interactions are being extracted.
    :param target_text: Prescribing information / label text for the drug.
    :returns: Raw model output string starting with ``[``
    """
    # Build the prompt string
    prompt = 'SYSTEM: You are a medical informatics expert. Extract drug-drug interactions into a structured JSON list. The list should follow the following format: [{"@type": "","@precipitant": "","@precipitantCode": "","@effect": ""},...].\n\n'
    prompt += f"### TASK\nDRUG: {target_drug}\nTEXT:\n{target_text}\n\nRESULT:"

    # Execute – force JSON array start
    formatted_input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n["
    outputs = llm.generate([formatted_input], extract_params)
    return "[" + outputs[0].outputs[0].text

def medgemma_base_prompt(prompt: str, force_model_to_start_with: str = "") -> str:
    """
    Run a single-turn instruction prompt through MedGemma and return the
    generated text.

    The model response can optionally be *seeded* by providing a prefix via
    ``force_model_to_start_with``, which constrains the decoding to continue
    from that string (useful for JSON mode).

    :param prompt: The user-facing instruction or question.
    :param force_model_to_start_with: Optional prefix prepended to the model
        turn before generation (such as ``"{"`` to force JSON output).
    :returns: The generated text, with ``force_model_to_start_with`` prepended.
    """
    formatted_input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{force_model_to_start_with}"
    outputs = llm.generate([formatted_input], base_params)
    return force_model_to_start_with + outputs[0].outputs[0].text