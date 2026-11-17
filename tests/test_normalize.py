from src.normalize import normalize_text

def test_nfc_and_whitespace_only():
    assert normalize_text("  नमस्ते\u00a0\n दुनिया  ") == "नमस्ते दुनिया"
    assert normalize_text("abc! 123") == "abc! 123"
