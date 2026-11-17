#!/usr/bin/env bash
# Full remote run. This is intentionally the only script that installs heavy packages or trains models.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -x .venv/bin/python ]]; then "$PYTHON_BIN" -m venv .venv; fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/prepare_data.py --input AnnotationFinal.csv --output data
.venv/bin/python -m pytest -q
for model in muril xlmr hybrid; do
  .venv/bin/python -m src.train --config "configs/${model}.yaml"
  for seed in 13 42 2024; do
    .venv/bin/python scripts/extract_rationales.py --config "configs/${model}.yaml" --checkpoint "outputs/metrics/${model}_seed${seed}.pt" --output outputs/rationales
    .venv/bin/python scripts/evaluate_faithfulness.py --config "configs/${model}.yaml" --checkpoint "outputs/metrics/${model}_seed${seed}.pt" --rationales outputs/rationales --output outputs/metrics
  done
done
.venv/bin/python scripts/make_html.py --rationales outputs/rationales --output outputs/figures
.venv/bin/python scripts/build_tables.py
echo "Completed. Metrics: outputs/metrics; rationales: outputs/rationales; viewer: outputs/figures"
