"""
BAYMAX AI – Medical Document & PDF Parser
==========================================
Extracts text, metadata, and structured clinical fields from medical documents
and PDFs using PyMuPDF (fitz) and regex-based clinical entity extraction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from app.utils.logger import get_logger

log = get_logger(__name__)


class MedicalDocumentParser:
    """
    Parser for clinical documents, lab reports, and prescriptions in PDF or image format.
    """

    def parse_document_bytes(
        self,
        file_bytes: bytes,
        filename: str = "document.pdf",
    ) -> Dict[str, Any]:
        """
        Parse raw bytes from a PDF or image into clinical text and structured fields.
        """
        filename_lower = filename.lower()
        extracted_text = ""

        if filename_lower.endswith(".pdf") or file_bytes[:4] == b"%PDF":
            extracted_text = self._extract_from_pdf(file_bytes)
        else:
            extracted_text = self._extract_from_image_or_text(file_bytes, filename)

        if not extracted_text.strip():
            return {
                "available": True,
                "raw_text": "",
                "document_type": "Unknown Document",
                "extracted_fields": {},
                "overall_confidence": 0.0,
            }

        # Clean extracted text
        clean_text = self._clean_text(extracted_text)

        # Detect document type
        doc_type = self._detect_document_type(clean_text)

        # Extract structured medical fields
        fields = self._extract_medical_fields(clean_text)

        confidence = 0.95 if len(clean_text) > 100 else 0.80

        log.info(
            "Medical document parsed | type='{}' | length={} | fields={}",
            doc_type, len(clean_text), len(fields),
        )

        return {
            "available": True,
            "raw_text": clean_text,
            "document_type": doc_type,
            "extracted_fields": fields,
            "overall_confidence": confidence,
            "provider": "pymupdf_clinical_parser",
        }

    def _extract_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text across all pages of a PDF using PyMuPDF."""
        try:
            import pymupdf as fitz
        except ImportError:
            try:
                import fitz
            except ImportError:
                log.error("PyMuPDF / fitz is not installed")
                return ""

        text_parts: List[str] = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_text = page.get_text("text")
                if page_text and page_text.strip():
                    text_parts.append(f"--- Page {page_idx + 1} ---\n{page_text.strip()}")
            doc.close()
        except Exception as exc:
            log.warning("PyMuPDF text extraction failed: {}", exc)

        return "\n\n".join(text_parts)

    def _extract_from_image_or_text(self, file_bytes: bytes, filename: str) -> str:
        """Extract text from text-based or image-based files."""
        # Check if plain text
        try:
            text = file_bytes.decode("utf-8")
            if len(text.strip()) > 5:
                return text
        except UnicodeDecodeError:
            pass

        # Try OCR via pytesseract if installed
        try:
            import io
            from PIL import Image
            import pytesseract

            img = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(img)
            if text and text.strip():
                return text
        except Exception as ocr_err:
            log.debug("Tesseract OCR unavailable or failed: {}", ocr_err)

        return ""

    def _clean_text(self, text: str) -> str:
        """Normalize line breaks and excessive whitespace."""
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _detect_document_type(self, text: str) -> str:
        """Classify document based on clinical keyword presence."""
        t_lower = text.lower()

        if any(w in t_lower for w in ["prescription", "rx", "sig:", "dispense", "refill", "dosage", "tablet", "capsule", "syrup"]):
            return "Medical Prescription"
        elif any(w in t_lower for w in ["lab report", "test report", "pathology", "biochemistry", "hematology", "reference range", "normal range", "specimen"]):
            return "Laboratory & Diagnostic Report"
        elif any(w in t_lower for w in ["discharge summary", "inpatient", "admission date", "discharge date", "hospital summary"]):
            return "Discharge Summary"
        elif any(w in t_lower for w in ["consultation", "patient history", "clinical note", "doctor's advice", "follow up"]):
            return "Clinical Consultation Note"
        return "Medical Document"

    def _extract_medical_fields(self, text: str) -> Dict[str, str]:
        """Extract key medical entities like Patient Name, Date, Doctor, Diagnosis, and Meds."""
        fields: Dict[str, str] = {}
        lines = text.splitlines()

        # 1. Patient Name
        for line in lines:
            m = re.search(r"(?:patient\s*(?:name)?|name\s*(?:of patient)?)\s*[:\-]\s*([A-Za-z\s.]+)", line, re.IGNORECASE)
            if m:
                fields["patient_name"] = m.group(1).strip()
                break

        # 2. Date
        for line in lines:
            m = re.search(r"(?:date|dated|collected\s*on)\s*[:\-]\s*([0-9]{1,4}[\/\-\.][0-9]{1,2}[\/\-\.][0-9]{1,4})", line, re.IGNORECASE)
            if m:
                fields["date"] = m.group(1).strip()
                break

        # 3. Doctor / Physician
        for line in lines:
            m = re.search(r"(?:dr\.|doctor|physician|consultant)\s*[:\-]?\s*([A-Za-z\s.,]+)", line, re.IGNORECASE)
            if m:
                fields["doctor"] = m.group(0).strip()
                break

        # 4. Diagnosis / Impression
        for line in lines:
            m = re.search(r"(?:diagnosis|impression|findings|chief complaint)\s*[:\-]\s*([^\n\r]+)", line, re.IGNORECASE)
            if m:
                fields["diagnosis"] = m.group(1).strip()
                break

        # 5. Medications / Rx
        meds = []
        for line in lines:
            m = re.search(r"(?:tab(?:let)?|cap(?:sule)?|syp|inj|syrup)\.?\s+([A-Za-z0-9\-]+(?:\s+[0-9]+(?:\s*mg|\s*ml)?)?)", line, re.IGNORECASE)
            if m:
                meds.append(m.group(0).strip())
        if meds:
            fields["medications"] = ", ".join(meds[:5])

        return fields


medical_document_parser = MedicalDocumentParser()
