"""
Exposes ``POST /in-context/`` for few-shot drug-drug interaction extraction.

Given a drug name the endpoint:

1. Queries the DailyMed API for the drug's prescribing label (Section 7).
2. Uses cached example labels as few-shot demonstrations in the MedGemma prompt.
3. Returns a JSON array of structured interaction objects.

If DailyMed has no record for the drug, an error dict is returned.
"""

from fastapi import APIRouter
from utils.req_res_structures import BaseRequest

from inference.medgemma_in_context import extract_interactions_from_drug

router = APIRouter(prefix="/in-context", tags=["in-context"])

@router.post("/")
async def prompt_in_context(input: BaseRequest):
    """
    Run the full DailyMed-fetch and few-shot-extraction pipeline.

    :param input: :class:`~utils.req_res_structures.BaseRequest` where
        ``prompt`` is the drug name to look up.
    :returns: JSON array string of interaction objects, or
              ``{"error": "<message>"}`` if the label is unavailable.
    """
    return extract_interactions_from_drug(input.prompt)
