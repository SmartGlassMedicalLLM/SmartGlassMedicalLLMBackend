import fitz

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    return full_text