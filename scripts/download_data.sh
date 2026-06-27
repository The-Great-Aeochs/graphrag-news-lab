#!/usr/bin/env bash
# Download the MultiHop-RAG corpus + QA pairs from HuggingFace into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
BASE="https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main"
echo "downloading corpus.json ..."
curl -sL -o data/raw/corpus.json "$BASE/corpus.json"
echo "downloading MultiHopRAG.json ..."
curl -sL -o data/raw/MultiHopRAG.json "$BASE/MultiHopRAG.json"
echo "done:"
ls -la data/raw/
