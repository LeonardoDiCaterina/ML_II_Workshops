#!/usr/bin/env bash
set -e

echo "=== ML II Workshops Environment Setup ==="
if command -v conda &> /dev/null; then
    echo "Found conda. Creating 'ml2_workshops' environment..."
    conda env create -f environment.yml || conda env update -f environment.yml --prune
    echo "To activate: conda activate ml2_workshops"
else
    echo "Conda not found. Falling back to virtualenv with pip..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e ".[test]"
    echo "To activate: source .venv/bin/activate"
fi
