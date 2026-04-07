"""
Lightweight helpers used across the project.
"""

import fitz

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all plain text from a PDF file on disk.

    Uses PyMuPDF (``fitz``) to iterate every page and concatenate the
    extracted text. The document is closed after reading.

    :param pdf_path: Absolute or relative path to the ``.pdf`` file.
    :returns: A single string containing the full text of the document.
    """
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    return full_text