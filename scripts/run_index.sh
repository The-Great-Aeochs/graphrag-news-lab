#!/usr/bin/env bash
# Build both indexes: GraphRAG (graph + communities + reports) and the plain
# RAG baseline (Chroma). Requires a populated .env.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "==> 1/3 building subset (input/*.txt) ..."
python ingest/subset.py

echo "==> 2/3 GraphRAG index (this is the expensive, one-time LLM pass) ..."
graphrag index --root .

echo "==> 3/3 plain RAG baseline (Chroma) ..."
python -m baseline.plain_rag build

echo "done. now run:  python app.py   or   python -m eval.run_eval --limit 60"
