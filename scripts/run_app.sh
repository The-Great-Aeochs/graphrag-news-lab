#!/usr/bin/env bash
# Launch the GraphRAG News Lab Gradio app.
# Usage:  bash scripts/run_app.sh        (then open the printed URL)
#         Ctrl+C to stop.
set -euo pipefail
cd "$(dirname "$0")/.."

# Activate the project venv.
if [ ! -d .venv ]; then
  echo "No .venv found. Create it first:  uv venv --python 3.12 && uv pip install -e ."
  exit 1
fi
source .venv/bin/activate

# Sanity checks: key + built indexes.
if [ ! -f .env ] || grep -q "<API_KEY>" .env; then
  echo "⚠️  .env missing or key not set. Paste your OpenAI key into .env first."
  exit 1
fi
if [ ! -d output ] || [ ! -d .chroma ]; then
  echo "⚠️  Indexes not built. Run:  bash scripts/run_index.sh"
  exit 1
fi

# Free port 7860 if a previous instance is still running.
pkill -f "app.py" 2>/dev/null || true
sleep 1

echo "Starting app → open http://127.0.0.1:7860  (Ctrl+C to stop)"
python app.py
