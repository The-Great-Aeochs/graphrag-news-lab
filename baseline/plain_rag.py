"""Plain top-k RAG baseline -- the A/B control.

Deliberately minimal: chunk -> embed -> cosine top-k -> stuff context -> generate.
No query rewriting, no rerank, no graph. This is the pipeline Weeks 2-3 sharpened,
and the thing GraphRAG has to beat on sensemaking and multi-hop questions.

Build the index once:
    python -m baseline.plain_rag build

Then query programmatically via PlainRAG().answer(query).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import tiktoken

import chromadb

from common import (
    CHAT_MODEL,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    embed_texts,
    get_openai_client,
    load_corpus,
)

COLLECTION = "news_baseline"
_ENC = tiktoken.get_encoding("o200k_base")


def _chunk(text: str) -> list[str]:
    """Token-window chunking that mirrors GraphRAG's chunker settings."""
    toks = _ENC.encode(text)
    step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
    chunks = []
    for start in range(0, len(toks), step):
        window = toks[start : start + CHUNK_SIZE_TOKENS]
        if window:
            chunks.append(_ENC.decode(window))
        if start + CHUNK_SIZE_TOKENS >= len(toks):
            break
    return chunks


def build() -> None:
    """Chunk + embed the subset corpus into a persistent Chroma collection."""
    corpus = load_corpus()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Rebuild cleanly each time so the index always matches the current subset.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    for art in corpus:
        body = f"{art['title']}\n\n{art.get('body') or ''}"
        for i, ch in enumerate(_chunk(body)):
            ids.append(f"{art['doc_id']}::chunk_{i}")
            docs.append(ch)
            metas.append(
                {
                    "doc_id": art["doc_id"],
                    "title": art["title"],
                    "source": art["source"],
                    "published_at": art["published_at"],
                }
            )

    print(f"embedding {len(docs)} chunks from {len(corpus)} articles with {EMBEDDING_MODEL} ...")
    # Embed in batches to stay under request limits.
    BATCH = 128
    for s in range(0, len(docs), BATCH):
        batch = docs[s : s + BATCH]
        embs = embed_texts(batch)
        col.add(
            ids=ids[s : s + BATCH],
            documents=batch,
            embeddings=embs,
            metadatas=metas[s : s + BATCH],
        )
        print(f"  indexed {min(s + BATCH, len(docs))}/{len(docs)}")
    print(f"done. collection '{COLLECTION}' has {col.count()} chunks.")


@dataclass
class PlainResult:
    response: str
    retrieved_doc_ids: list[str]
    retrieved_chunks: list[dict] = field(default_factory=list)


_SYSTEM = (
    "You answer strictly from the provided news excerpts. If the excerpts do not "
    "contain the answer, say you don't have enough information -- do not guess. "
    "Be concise and cite the article titles you used."
)


class PlainRAG:
    def __init__(self, top_k: int = 8):
        self.top_k = top_k
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = self.client.get_collection(COLLECTION)
        self.llm = get_openai_client()

    def retrieve(self, query: str) -> list[dict]:
        q_emb = embed_texts([query])[0]
        res = self.col.query(query_embeddings=[q_emb], n_results=self.top_k)
        out = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append({"text": doc, **meta, "distance": dist})
        return out

    def answer(self, query: str) -> PlainResult:
        chunks = self.retrieve(query)
        context = "\n\n---\n\n".join(
            f"[{c['title']} | {c['source']} | {c['published_at'][:10]}]\n{c['text']}"
            for c in chunks
        )
        msg = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"News excerpts:\n\n{context}\n\nQuestion: {query}"},
        ]
        resp = self.llm.chat.completions.create(
            model=CHAT_MODEL, messages=msg, temperature=0.0
        )
        # Dedupe retrieved doc ids preserving rank order (for Recall@K eval).
        seen, doc_ids = set(), []
        for c in chunks:
            if c["doc_id"] not in seen:
                seen.add(c["doc_id"])
                doc_ids.append(c["doc_id"])
        return PlainResult(
            response=resp.choices[0].message.content,
            retrieved_doc_ids=doc_ids,
            retrieved_chunks=chunks,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    else:
        print("usage: python -m baseline.plain_rag build")
