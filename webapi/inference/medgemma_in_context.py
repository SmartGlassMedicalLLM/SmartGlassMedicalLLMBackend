import json
import glob
import os
from webapi.inference.medgemma_utils import llm, extract_params
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

## Build in-context prompt

def run_in_context_extraction(target_drug, target_text, shots=2):
    # Select examples for the context
    context_examples = [ex for ex in drug_data if ex['drug'] != target_drug][:shots]

    # Build the prompt string
    sys_prompt = "You are a medical informatics expert. Extract drug-drug interactions into a structured JSON list.\n\n"

    formatted_input = "" # https://ai.google.dev/gemma/docs/core/prompt-structure
    for i, ex in enumerate(context_examples):
        formatted_input += "<start_of_turn>user\n"

        if i == 0:
            formatted_input += sys_prompt

        formatted_input += f"Based on the following prescribing information, provide the drug-drug interactions for the drug, {ex['drug']}, in a sturctured JSON list.\n\n"
        formatted_input += ex['text']

        formatted_input += "<end_of_turn>\n<start_of_turn>model\n"
        formatted_input += json.dumps(ex['interactions']) + "<end_of_turn>\n"

    # Adds the actual question/prompt
    formatted_input += f"<start_of_turn>user\nBased on the following prescribing information, provide the drug-drug interactions for the drug, {target_drug}, in a sturctured JSON list.\n\n"
    formatted_input += target_text
    formatted_input += "<end_of_turn>\n<start_of_turn>model\n[" # Force JSON start

    # Execute
    outputs = llm.generate([formatted_input], extract_params)
    return outputs[0].outputs[0].text

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