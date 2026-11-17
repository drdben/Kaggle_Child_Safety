import ast
from pathlib import Path


def test_class_weights_are_explicit_float32():
    """Prevent a regression to float64 class weights in CrossEntropyLoss."""
    source = Path("src/train.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "dtype=torch.float32" in source
