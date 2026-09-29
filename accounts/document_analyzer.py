import re
import easyocr


# ---------------------------------------------------------
# EasyOCR Reader
# ---------------------------------------------------------

_reader = None


def get_reader():
    global _reader

    if _reader is None:
        _reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

    return _reader


# ---------------------------------------------------------
# Generic OCR
# ---------------------------------------------------------

def extract_text(image_path):

    reader = get_reader()

    results = reader.readtext(
        image_path,
        detail=1
    )

    extracted_text = []
    confidence_values = []

    for result in results:

        if len(result) >= 3:

            text = result[1]
            confidence = result[2]

            extracted_text.append(text)
            confidence_values.append(confidence)

    text = " ".join(extracted_text)

    if confidence_values:

        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        ) * 100

    else:

        average_confidence = 0

    return text, round(average_confidence, 2)


# ---------------------------------------------------------
# Aadhaar Analysis
# ---------------------------------------------------------

def analyze_aadhaar(image_path):

    text, confidence = extract_text(
        image_path
    )

    normalized_text = text.replace(
        " ",
        ""
    )

    # Find 12 digit Aadhaar-like number
    aadhaar_matches = re.findall(
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        text
    )

    aadhaar_number = ""

    if aadhaar_matches:

        aadhaar_number = re.sub(
            r"\D",
            "",
            aadhaar_matches[0]
        )

    # Detect common Aadhaar indicators
    indicators = [
        "aadhaar",
        "uidai",
        "government",
        "india"
    ]

    indicator_count = 0

    lower_text = text.lower()

    for indicator in indicators:

        if indicator in lower_text:

            indicator_count += 1

    document_detected = (
        len(aadhaar_number) == 12
        or indicator_count >= 1
    )

    last_four = ""

    if len(aadhaar_number) == 12:

        last_four = aadhaar_number[-4:]

    return {

        "document_detected":
            document_detected,

        "aadhaar_number":
            aadhaar_number,

        "aadhaar_last_four":
            last_four,

        "confidence":
            confidence,

        "raw_text":
            text
    }


# ---------------------------------------------------------
# PAN Analysis
# ---------------------------------------------------------

def analyze_pan(image_path):

    text, confidence = extract_text(
        image_path
    )

    # Standard PAN format:
    # ABCDE1234F

    pan_matches = re.findall(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        text.upper()
    )

    pan_number = ""

    if pan_matches:

        pan_number = pan_matches[0]

    lower_text = text.lower()

    pan_indicators = [
        "income tax",
        "permanent account number",
        "pan"
    ]

    indicator_count = 0

    for indicator in pan_indicators:

        if indicator in lower_text:

            indicator_count += 1

    document_detected = (
        bool(pan_number)
        or indicator_count >= 1
    )

    return {

        "document_detected":
            document_detected,

        "pan_number":
            pan_number,

        "confidence":
            confidence,

        "raw_text":
            text
    }