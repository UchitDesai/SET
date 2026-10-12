import os
from typing import Optional
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


# Initialize OCR model once (important for performance)
ocr_model = ocr_predictor(pretrained=True)


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from PDF or Image using DocTR OCR.
    """

    if not os.path.exists(file_path):
        return ""

    if file_path.lower().endswith(".pdf"):
        doc = DocumentFile.from_pdf(file_path)
    else:
        doc = DocumentFile.from_images(file_path)

    result = ocr_model(doc)

    extracted_text = ""

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join(word.value for word in line.words)
                extracted_text += line_text + "\n"

    return extracted_text.strip()


def combine_inputs(
    raw_text: Optional[str] = None,
    url: Optional[str] = None,
    email_subject: Optional[str] = None,
    email_sender: Optional[str] = None,
    file_text: Optional[str] = None
) -> str:
    """
    Combine multimodal inputs into a single text blob.
    """

    combined_text = ""

    if email_subject:
        combined_text += f"Subject: {email_subject}\n"

    if email_sender:
        combined_text += f"Sender: {email_sender}\n"

    if raw_text:
        combined_text += raw_text + "\n"

    if url:
        combined_text += f"URL: {url}\n"

    if file_text:
        combined_text += file_text + "\n"

    return combined_text.strip()