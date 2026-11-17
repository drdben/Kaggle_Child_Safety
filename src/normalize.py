"""Conservative text normalization used consistently for splits and models."""
import re
import unicodedata


def normalize_text(text: object) -> str:
    """Apply NFC and collapse whitespace; deliberately retain punctuation and digits."""
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
