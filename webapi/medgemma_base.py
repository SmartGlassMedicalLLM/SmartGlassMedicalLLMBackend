import json
import glob
import os
import torch
from medgemma_utils import tokenizer, model

## Extraction build prompt

def run_extraction(target_drug, target_text):
    # Build the prompt string
    prompt = """SYSTEM: You are a medical informatics expert. Extract drug-drug interactions into a structured JSON list. The list should follow the following format: [{"@type": "","@precipitant": "","@precipitantCode": "","@effect": ""},...].\n\n"""
    prompt += f"### TASK\nDRUG: {target_drug}\nTEXT:\n{target_text}\n\nRESULT:"

    # Execute
    formatted_input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n[" # Force JSON start
    inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.01,
            do_sample=False
        )

    # Extract and format response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    final_json = "[" + response

    return final_json

## Basic prompt

def medgemma_base_prompt(prompt):
    formatted_input = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.01,
            do_sample=False
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response