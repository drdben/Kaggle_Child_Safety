from src.attribution.spans import select_words,merge_spans

def test_only_positive_and_adjacent_merge():
    words=[{"word":"a","char_start":0,"char_end":1,"score":.8},{"word":"b","char_start":2,"char_end":3,"score":.7},{"word":"c","char_start":4,"char_end":5,"score":-.9}]
    selected=select_words(words,"relative",.5)
    assert selected == [0,1]
    assert merge_spans("a b c",words,selected) == [{"text":"a b","start":0,"end":3,"score":1.5}]
