#!/usr/bin/env python3
"""No-download, no-GPU check for a fresh local clone and its input CSV."""
import csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.normalize import normalize_text
from src.attribution.align import subtoken_to_words
from src.attribution.spans import select_words,merge_spans

rows=list(csv.DictReader(open("AnnotationFinal.csv",encoding="utf-8",newline="")))
assert rows and {"Text","Class","VideoFile"} <= rows[0].keys()
assert normalize_text("क\u093f\t  test\n") == "कि test"
text="नमस्ते, बुरा शब्द"; words=subtoken_to_words(text,[(0,0),(0,6),(6,7),(8,12),(13,17),(0,0)],[0,.2,.1,.9,.4,0])
words[1]["score"]=.9; selected=select_words(words,"top1",1); spans=merge_spans(text,words,selected)
assert spans[0]["text"] == "बुरा"
print(f"Smoke test passed: {len(rows)} rows, normalization/alignment/span utilities are usable.")
