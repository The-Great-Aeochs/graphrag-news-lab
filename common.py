"""Shared config, paths, and OpenAI helpers used across all pipelines.

Keeping the embedding + chat model definitions in one place is what makes the
A/B comparison honest: the plain-RAG baseline and GraphRAG both embed with the
same model, so any difference in answers comes from the *architecture*, not the
vectors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- Paths -----------------------------------------------------------------
DATA = ROOT / "data"
SUBSET = DATA / "subset"
CORPUS_SUBSET = SUBSET / "corpus_subset.json"
QA_SUBSET = SUBSET / "qa_subset.json"
INPUT_DIR = ROOT / "input"
GRAPHRAG_OUTPUT = ROOT / "output"
CHROMA_DIR = ROOT / ".chroma"

# --- Models (single source of truth) ---------------------------------------
CHAT_MODEL = os.getenv("GRAPHRAG_LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small")

# Baseline chunking -- intentionally mirrors GraphRAG's settings.yaml chunker
# (1200-token chunks, 100 overlap) so neither side gets an unfair chunking edge.
CHUNK_SIZE_TOKENS = 1200
CHUNK_OVERLAP_TOKENS = 100


def get_openai_client():
    """Lazily build an OpenAI client. Raises a clear error if the key is unset."""
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY") or os.getenv("GRAPHRAG_API_KEY")
    if not key or key.startswith("<"):
        raise RuntimeError(
            "No OpenAI key found. Copy .env.example to .env and paste your key "
            "into OPENAI_API_KEY (and GRAPHRAG_API_KEY)."
        )
    return OpenAI(api_key=key)


def load_corpus() -> list[dict]:
    return json.loads(CORPUS_SUBSET.read_text())


def load_qa() -> list[dict]:
    return json.loads(QA_SUBSET.read_text())


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the shared embedding model."""
    client = get_openai_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]
