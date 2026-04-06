from webapi.inference.medgemma_utils import llm, base_params, extract_params

## Extraction build prompt

def run_extraction(target_drug, target_text):
    # Build the prompt string
    prompt = """SYSTEM: You are a medical informatics expert. Extract drug-drug interactions into a structured JSON list. The list should follow the following format: [{"@type": "","@precipitant": "","@precipitantCode": "","@effect": ""},...].\n\n"""
    prompt += f"### TASK\nDRUG: {target_drug}\nTEXT:\n{target_text}\n\nRESULT:"

    # Execute
    formatted_input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n[" # Force JSON start
    outputs = llm.generate([formatted_input], extract_params)
    return "[" + outputs[0].outputs[0].text

## Basic prompt

def medgemma_base_prompt(prompt, force_model_to_start_with = ""):
    formatted_input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{force_model_to_start_with}"
    outputs = llm.generate([formatted_input], base_params)
    return force_model_to_start_with + outputs[0].outputs[0].text