# GraphRAG News Lab

A focused showcase of **when GraphRAG beats plain RAG — and when it doesn't.**

One question is sent to **three architectures** side by side over the same news
corpus, and a ground-truth eval harness measures who wins on which kind of
question:

| Route | What it's for | The pitfall it fixes |
|---|---|---|
| **Plain top-k RAG** (baseline) | the control — chunk → embed → top-k → generate | — |
| **GraphRAG global search** | sensemaking: "what are the themes across the corpus?" | answer exists in *no single chunk* |
| **GraphRAG local search** | entity-centric: "what is X connected to?" | flat retrieval has *no representation of "between"* |

The discipline this repo is built to demonstrate: **reach for GraphRAG only when
you have a question your vector pipeline structurally cannot answer.** Most of the
time, plain RAG wins — and the eval is there to prove which case you're in.

## Dataset

[MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) (Tang & Yang, 2024) — a
news corpus that ships with multi-hop QA pairs **and ground-truth evidence
documents**. We index a coherent ~100-article **technology** subset (AI labs, the
FTX/crypto saga, Big Tech) and keep the 700+ QA pairs whose evidence stays fully
inside the subset, so retrieval can be scored against truth for free.

Each QA pair carries a `question_type` — `inference`, `comparison`, `temporal`,
or `null` — which lets the eval slice results by reasoning type. The headline
result to look for: plain RAG sags on comparison/temporal/multi-hop questions
while GraphRAG holds, and `null` questions test graceful abstention ("fail, don't
hallucinate").

## Architecture

```
corpus.json + MultiHopRAG.json
        │  ingest/subset.py  (100-article tech subset + intact QA)
        ▼
   input/*.txt
   ┌────┴─────────────────────────────┐
   ▼                                  ▼
GraphRAG index (settings.yaml)   Plain RAG baseline (baseline/plain_rag.py)
  chunk → extract entities/rels    chunk → embed → Chroma
  → Leiden communities → reports
   │ output/*.parquet                 │ .chroma/
   ▼                                  ▼
graphrag_app/query.py            baseline retriever
  global() / local()                 answer()
   └──────────────┬───────────────────┘
                  ▼
         harness/ab.py   →   app.py (Gradio, 3 panels)
                  ▼
         eval/run_eval.py  (Recall@K, MRR, accuracy by question_type)
```

Both pipelines embed with the **same** model (`text-embedding-3-small`) so the
A/B comparison is honest — only the architecture differs.

## Setup

```bash
# 1. install (uses uv; falls back to pip install -e . )
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .

# 2. data
bash scripts/download_data.sh

# 3. key — copy and paste your OpenAI key into both fields
cp .env.example .env   # then edit .env

# 4. build both indexes (GraphRAG index is the expensive one-time LLM pass)
bash scripts/run_index.sh
```

## Use

```bash
# interactive 3-way comparison
python app.py

# one question, all three routes, as JSON
python -m harness.ab "What are the main themes across this corpus?"

# ground-truth eval, sliced by question_type
python -m eval.run_eval --limit 60 --routes plain,graphrag_local
```

## Config notes

- **Models** (`settings.yaml`): `gpt-4o-mini` for extraction/reports/generation
  (keeps the per-chunk + per-community bill low), `text-embedding-3-small` for
  embeddings. Bump the chat model if extraction quality — which sets GraphRAG's
  whole ceiling — proves too noisy.
- **Entity types** are retuned for tech/business news
  (`organization, person, product, technology, event, location, financial_concept`)
  rather than GraphRAG's news-agnostic default — the single most important
  domain-tuning knob.
- **Scope/cost**: indexing runs an LLM over every chunk *and* every community at
  every hierarchy level. That's the deal GraphRAG makes — a heavy one-time index
  cost in exchange for answering questions plain RAG can't. If you don't have
  those questions, don't pay it.

## Honest limitations

- GraphRAG **local search is one-hop** — it gathers the entry entities'
  immediate neighbourhood and stops. Deep multi-hop chains ("who reports to the
  person who approved X") are a known weakness, not a feature.
- Local search's entry point is **pure vector similarity** over entity
  descriptions, so it inherits vector search's blind spots: phrase a query far
  from how an entity was described and the right entity can miss the top-k.
- Entity merging keys on exact title+type, so there's **no entity
  disambiguation** ("Jon" vs "Jon Márquez" stay separate nodes). Extraction
  quality is the performance ceiling.

## Sources

- Edge et al. (2024), *From Local to Global: A Graph RAG Approach to
  Query-Focused Summarization* — arXiv:2404.16130
- Tang & Yang (2024), *MultiHop-RAG: Benchmarking Retrieval-Augmented Generation
  for Multi-Hop Queries* — arXiv:2401.15391
- [microsoft/graphrag](https://github.com/microsoft/graphrag) — the indexing +
  query implementation this repo builds on
