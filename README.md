# ChildSafeDub: attribution-based rationale extraction from sentence-level supervision

This project trains Hindi sentence-level profanity classifiers (MuRIL, XLM-R, and a late-fusion MuRIL+XLM-R hybrid), then derives signed word-level rationales with Integrated Gradients. It does **not** train a token tagger and makes no span-F1 claim: ChildSafeDub has sentence labels, not gold spans.

## Local smoke test

The checked-in `.venv` is intentionally ignored. Create it and run the no-download check:

```bash
python3 -m venv .venv
.venv/bin/python scripts/smoke_test.py
```

This confirms the supplied CSV schema plus NFC/whitespace normalization, Devanagari offset handling, and span merging. It does not install PyTorch or download models.

## Remote full run

Clone this repository on the GPU machine, then run:

```bash
git pull origin main
bash scripts/run_all.sh
```

The script creates a venv if needed, installs `requirements.txt`, saves fixed 70/15/15 split IDs under `data/`, runs tests, trains each model for seeds 13/42/2024, runs IG on predicted-profane test items, and emits static HTML. Outputs are intentionally gitignored. Set `PYTHON_BIN` to select the remote Python executable. Each YAML config controls batch size, epochs (default 3; safe to increase to 5), precision-related environment choice, IG steps (50), aggregation, and hybrid mixing weight.

`prepare_data.py` detects explicit source/episode/cartoon-style columns and uses grouped splitting only when available; otherwise it uses stratified random splitting. `VideoFile` is never used as a feature or grouping proxy. Class-weighted cross entropy, macro-F1 checkpoint selection, early stopping, ROC-AUC, confusion matrices, per-class reports, config metadata, and the git hash are retained per run.

## Outputs and interpretation

Classification results are in `outputs/metrics/`; rationale JSONL is in `outputs/rationales/`; HTML viewers are in `outputs/figures/`. A rationale records original character offsets, signed normalized word scores, convergence delta, and the requested threshold/top-k/top-1 spans. Tokenizer offsets map only to whitespace words; punctuation is stripped only in the presentation field.

Use faithfulness statistics (comprehensiveness, sufficiency, flip rate, random controls) to compare extraction strategies, not as span accuracy. If `data/lexicon.txt` is later supplied, any overlap result must be labelled *lexicon-proxy, not gold*.

## Reproducibility and run order

The intended gated order is data prep/tests → MuRIL plus attribution/alignment review → XLM-R → hybrid → faithfulness/tables. Stop after a model if validation/test macro-F1 is materially below ~0.85 or the 20-example alignment audit fails. Never tune extraction thresholds on test: select the preferred strategy on validation, then report it on test.
