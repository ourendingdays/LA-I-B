#!/usr/bin/env bash
# run_app.sh — Run streamlit app in a virtual environment.
# Run from the repo root. Idempotent: safe to re-run.

echo "Activating virtual environment..."
source venv/bin/activate

echo "Running streamlit app..."
streamlit run frontend/app.py 