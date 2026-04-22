"""
Extracts drug-drug interactions from DailyMed prescribing information using
few-shot (in-context) prompting with MedGemma.
"""

import json
import glob
import os
from inference.medgemma_utils import llm, extract_params
import requests
from xml.etree import ElementTree

def load_drug_data(directory: str) -> list[dict]:
    """
    Load all drug interaction JSON samples from a directory.

    Each file must contain a JSON object with at least the keys ``"drug"``,
    ``"text"``, and ``"interactions"``.  Files that fail to parse are silently
    skipped.

    :param directory: Path to the folder containing ``.json`` example files.
    :returns: List of parsed drug data dicts or empty list on failure
    """
    all_data = []
    file_paths = glob.glob(os.path.join(directory, "*.json"))
    for path in file_paths:
        with open(path, 'r') as f:
            try:
                item = json.load(f)
                all_data.append(item)
            except json.JSONDecodeError:
                continue
    return all_data

dir = os.path.dirname(os.path.realpath(__file__))
drug_data = load_drug_data(os.path.join(dir, "drug_extraction_samples"))
"""Pre-loaded examples used by :func:`run_in_context_extraction`."""

def run_in_context_extraction(target_drug: str, target_text: str, shots: int = 2) -> str:
    """
    Build a multi-shot prompt from cached examples and run structured
    interaction extraction with MedGemma.

    Example drugs that match ``target_drug`` are excluded from the shot pool
    to avoid data leakage.  The prompt follows the Gemma chat turn format
    (``<start_of_turn>`` / ``<end_of_turn>``), and the model is forced to start
    with ``[`` for a JSON array response.

    :param target_drug: Name of the drug to extract interactions for.
    :param target_text: Prescribing information text.
    :param shots: Number of few-shot examples to include (default 2).
    :returns: Raw model output string that *should* be valid JSON.
    """
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

def fetch_setId(drug_name: str) -> str | None:
    """
    Retrieve the SET ID for the first matching drug label from the DailyMed API.

    Queries ``/dailymed/services/v2/spls.json`` with the drug name and returns
    the ``setid`` of the first result.

    :param drug_name: Common or brand name of the drug (case-insensitive).
    :returns: The DailyMed SET ID string, or ``None`` if no results are found
              or the request fails.
    """
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

def fetch_dailyMed_interactions(drug_name: str) -> dict | None:
    """
    Retrieve Section 7 (Drug Interactions) from the DailyMed XML label for a
    given drug, identified via :func:`fetch_setId`.

    Parses the HL7 V3 XML for the ``34073-7`` LOINC section code and strips
    all text content.

    :param drug_name: Common or brand name of the drug.
    :returns: Dict with keys ``"drug"`` (uppercased), ``"text"`` (plain-text
              interaction section), and ``"source"`` (``"DailyMed API"``);
              or ``None`` if the SET ID cannot be found, the request fails, or
              Section 7 is absent from the label.
    """
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

def extract_interactions_from_drug(drug_name: str) -> str | dict:
    """
    Fetch DailyMed data for a drug and return a JSON
    array of extracted drug-drug interactions.

    Calls :func:`fetch_dailyMed_interactions` to get Section 7 text, then
    passes it to :func:`run_in_context_extraction` for LLM inference.

    :param drug_name: Common or brand name of the drug to look up.
    :returns: A JSON array string on success, or a dict
              ``{"error": "<message>"}`` if the drug label cannot be found.
    """
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