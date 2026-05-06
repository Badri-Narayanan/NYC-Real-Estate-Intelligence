#!/usr/bin/env bash

set -e

PYTHON=${PYTHON:-python3}

echo "==> Creating virtual environment ./venv"
$PYTHON -m venv venv

echo "==> Activating venv"
# shellcheck disable=SC1091
source venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing requirements"
pip install -r requirements.txt

echo "==> Setting up .env"
if [ ! -f config/.env ]; then
    cp config/.env.example config/.env
    echo "    Created config/.env (please add your ANTHROPIC_API_KEY)"
fi

echo "==> Done."
echo
echo "Next steps:"
echo "  1. Edit config/.env and set ANTHROPIC_API_KEY=your-key"
echo "  2. python main.py --step all      # Run full pipeline"
echo "  3. streamlit run app/streamlit_app.py"
