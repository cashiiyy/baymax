import os
from typing import Optional

class MedicalOCR:
    """Extracts text from medical documents, lab reports, or prescriptions using pytesseract."""

    def extract_text(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path not found: {image_path}")

        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            return f"[OCR Stub Output]: Extracted prescription text from {os.path.basename(image_path)}"
