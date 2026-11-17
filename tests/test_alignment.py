import pytest
from src.attribution.align import subtoken_to_words

@pytest.mark.parametrize("model",["google/muril-base-cased","xlm-roberta-base"])
def test_devanagari_offsets_fast_tokenizer(model):
    transformers=pytest.importorskip("transformers")
    try: tok=transformers.AutoTokenizer.from_pretrained(model,use_fast=True)
    except OSError: pytest.skip("Tokenizer unavailable; run this integration test on the remote machine")
    text="कि\u0902तु गाली!" # combining marks must remain inside the first whitespace word
    e=tok(text,return_offsets_mapping=True); words=subtoken_to_words(text,e["offset_mapping"],[1.0]*len(e["input_ids"]))
    assert [w["word"] for w in words] == ["किंतु","गाली!"]
    assert words[0]["char_start"] == 0 and words[1]["char_start"] == 6
