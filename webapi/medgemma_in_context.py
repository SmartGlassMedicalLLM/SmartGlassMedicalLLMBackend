import json
import glob
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import requests
from xml.etree import ElementTree

## Collect drug data

def load_drug_data(directory):
    all_data = []
    file_paths = glob.glob(os.path.join(directory, "*.json"))
    for path in file_paths:
        with open(path, 'r') as f:
            try:
                item = json.load(f)
                all_data.append(item)
            except: continue
    return all_data

drug_data = load_drug_data("/content/trainingFilesConverted/")

## Get model and tokenizer

model_id = "google/medgemma-4b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

## Build in-context prompt

def run_in_context_extraction(target_drug, target_text, shots=2):
    # Select examples for the context
    context_examples = [ex for ex in drug_data if ex['drug'] != target_drug][:shots]

    # Build the prompt string
    prompt = "SYSTEM: You are a medical informatics expert. Extract drug-drug interactions into a structured JSON list.\n\n"

    for i, ex in enumerate(context_examples):
        prompt += f"### EXAMPLE {i+1}\n"
        prompt += f"DRUG: {ex['drug']}\n"
        prompt += f"TEXT: {ex['text'][:600]}...\n" # Truncated for token safety
        prompt += f"RESULT: {json.dumps(ex['interactions'])}\n\n"

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

## DailyMed API

def fetch_setId(drug_name):
    """
    Retrieves the SET ID for the first drug with the given name from the DailyMed API.
    """
    # API endpoint
    endpoint = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"

    # Build query
    params = {
        'drug_name': drug_name
    }

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        if 'data' in data:
            print(f"Found {len(data['data'])} results for {drug_name}.")
            print(f"Choosing index 0: {data['data'][0]['title']}")
            return data['data'][0]['setid']
        return None
    except Exception as e:
        print(f"Error fetching SET ID for {drug_name}: {str(e)}")
        return None

def fetch_dailyMed_interactions(drug_name):
    """
    Retrieves Section 7 (Drug Interactions) from the DailyMed API.
    """
    # Get the SET ID for the drug name
    setId = fetch_setId(drug_name)
    if setId is None:
        return None

    # API endpoint
    endpoint = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setId}.xml"

    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        tree = ElementTree.fromstring(response.content)

        section = tree.find(".//{urn:hl7-org:v3}code[@code='34073-7']/..")

        if section is not None:
            return {
                "drug": drug_name.upper(),
                "text": "".join(section.itertext()).strip(),
                "source": "DailyMed API"
            }
        return None
    except Exception as e:
        print(f"Error fetching {drug_name}: {str(e)}")
        return None

## Pull drug info with DailyMed and then prompt for extraction

def extract_interactions_from_drug(drug_name):
    collected_data = fetch_dailyMed_interactions(drug_name)

    if collected_data is None:
        print(f"Could not find a label for {drug_name}")
        return {
            "error": f"Could not find a label for {drug_name}"
        }

    final_json = run_in_context_extraction(
        target_drug=collected_data['drug'],
        target_text=collected_data['text']
    )

    return final_json