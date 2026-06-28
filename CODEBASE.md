# Codebase Walkthrough

A guide to explain this repo: the end-to-end flow first, then each file with its
key lines. Read top to bottom — it follows the data.

---

## 1. The flow (one picture)

```
data/raw/*.json                          (corpus + QA pairs, downloaded)
      │
      ▼  ingest/subset.py
input/*.txt + data/subset/*.json         (100-article tech subset + intact QA)
      │
      ├──────────────────────────┐
      ▼                          ▼
GraphRAG index             baseline/plain_rag.py build
(settings.yaml, CLI)       chunk → embed → Chroma
graph + communities             │
+ reports → output/*.parquet    │
      │                          │
      ▼ graphrag_app/query.py    ▼
  graph_global() / graph_local()   PlainRAG().answer()
      └─────────────┬──────────────┘
                    ▼  harness/ab.py   (one query → all 3 routes)
                    ├──► app.py        (Gradio: 3 panels)
                    └──► eval/run_eval.py (Recall@K, MRR, accuracy by question_type)
```

Three "routes" answer every question: **simple** (baseline), **graphrag_global**
(global search), **graphrag_local** (entity-centric). Everything below builds toward
making those three callable through one interface.

---

## 2. `common.py` — shared config 

Why it exists: both pipelines must use the **same embedding model**, or the A/B
comparison is not useful then.

```python
ROOT = Path(__file__).resolve().parent      # repo root
load_dotenv(ROOT / ".env")                   # load OPENAI_API_KEY etc.
```

```python
CHAT_MODEL      = os.getenv("GRAPHRAG_LLM_MODEL", "gpt-4o-mini")          # env-overridable
EMBEDDING_MODEL = os.getenv("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE_TOKENS = 1200; CHUNK_OVERLAP_TOKENS = 100   # mirrors settings.yaml chunker
```

```python
def get_openai_client():
    key = os.getenv("OPENAI_API_KEY") or os.getenv("GRAPHRAG_API_KEY")
    if not key or key.startswith("<"):       # placeholder check → clear error, not a crash
        raise RuntimeError(...)
    return OpenAI(api_key=key)               # lazy: only built when actually needed
```

```python
def embed_texts(texts):                      # one batched embedding call, shared by both sides
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]
```
`load_corpus()` / `load_qa()` just read the subset JSON. **Takeaway:** models, paths,
keys all flow from here.

---

## 3. `ingest/subset.py` — build the 100-article subset

Why: We focus on a subset right now. We take a coherent, eval-friendly slice.

```python
CATEGORY = "technology"; TARGET_ARTICLES = 100; N_NULL_QUERIES = 15
```

```python
for i, art in enumerate(corpus):
    art["doc_id"] = f"doc_{i:04d}"           # stable id per article (used everywhere later)
url2art = {a["url"]: a for a in corpus}      # evidence links to articles by URL
```

```python
ref_counts = Counter()                       # rank tech articles by how often QA cites them
for q in qa:
    for ev in q["evidence_list"]:
        if url2art.get(ev["url"], {}).get("category") == CATEGORY:
            ref_counts[ev["url"]] += 1
tech.sort(key=lambda a: ref_counts[a["url"]], reverse=True)
kept = tech[:TARGET_ARTICLES]                # the 100 most-referenced → preserves QA
```

```python
intact = [q for q in qa                      # keep a QA pair ONLY if every evidence
          if all(ev["url"] in kept_urls      # article is in the subset — else its
                 for ev in q["evidence_list"])]  # ground-truth answer is unreachable
```

```python
for q in qa_subset:                          # attach resolved doc_ids → used for Recall@K
    q["evidence_doc_ids"] = [url2art[ev["url"]]["doc_id"] for ev in q["evidence_list"] ...]
```

```python
header = f"Title: {art['title']}\nSource: {art['source']}\nPublished: ...\n\n"
(INPUT / f"{art['doc_id']}_{slugify(art['title'])}.txt").write_text(header + body)
```
The header gives the extractor publication context (helps temporal/grounding). Output:
`input/*.txt` for GraphRAG, plus `corpus_subset.json` / `qa_subset.json`.

---

## 4. GraphRAG index — `settings.yaml` (config)

Run by the CLI `graphrag index`. The three lines we customized:

```yaml
model: gpt-4o-mini                 # LLM for extraction + reports (many calls)
model: text-embedding-3-small      # SAME embedder as the baseline → honest A/B
entity_types: [organization,person,product,technology,event,location,financial_concept]
                                   # retuned from the news-agnostic default for tech/business
```
The CLI produces `output/*.parquet`: `entities`, `relationships`, `communities`,
`community_reports`, `text_units`, `documents`. These are the graph + the
pre-written community summaries that query time reads.

---

## 5. `baseline/plain_rag.py` — the control (plain top-k RAG)

Deliberately minimal: chunk → embed → cosine top-k → stuff → generate.

```python
def _chunk(text):                            # token-window chunking, mirrors GraphRAG
    toks = _ENC.encode(text)
    step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS    # 1200 - 100 = 1100 stride
    for start in range(0, len(toks), step):
        chunks.append(_ENC.decode(toks[start:start+CHUNK_SIZE_TOKENS]))
```

```python
def build():                                 # `python -m baseline.plain_rag build`
    client.delete_collection(COLLECTION)     # rebuild clean each time → matches current subset
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    for art in corpus:                        # chunk every article
        for i, ch in enumerate(_chunk(body)):
            ids.append(f"{art['doc_id']}::chunk_{i}")   # id carries doc_id → recoverable later
            metas.append({"doc_id": art["doc_id"], "title": ..., "source": ...})
    for s in range(0, len(docs), 128):        # embed in batches of 128
        col.add(ids=..., documents=..., embeddings=embed_texts(batch), metadatas=...)
```

```python
class PlainRAG:
    def retrieve(self, query):
        q_emb = embed_texts([query])[0]
        res = self.col.query(query_embeddings=[q_emb], n_results=self.top_k)  # top-8 cosine
        return [{"text": doc, **meta, "distance": dist} for ...]

    def answer(self, query):
        chunks = self.retrieve(query)
        context = "\n\n---\n\n".join(f"[{title}|{source}|{date}]\n{text}" for c in chunks)
        resp = self.llm.chat.completions.create(model=CHAT_MODEL, messages=[
            {"role":"system","content": _SYSTEM},        # "answer ONLY from excerpts, else abstain"
            {"role":"user","content": f"News excerpts:\n{context}\n\nQuestion: {query}"}],
            temperature=0.0)                              # precision task, not creative
        # dedupe retrieved doc_ids in rank order → for Recall@K eval
        return PlainResult(resp..., retrieved_doc_ids=doc_ids, retrieved_chunks=chunks)
```
`_SYSTEM` enforces abstention (the null-query / "don't hallucinate" test).
`temperature=0.0` is the lecture's "RAG is a precision task" rule.

---

## 6. `graphrag_app/query.py` — wrap GraphRAG global + local

Turns GraphRAG's API into the same `(response, doc_ids)` shape as the baseline.

```python
@lru_cache(maxsize=1)
def _config():
    return load_config(ROOT)                  # parse settings.yaml once, cache it

def _read(name):
    return pd.read_parquet(GRAPHRAG_OUTPUT / f"{name}.parquet")   # load an artifact
```

```python
def graph_global(query, response_type="multiple paragraphs"):
    response, context = asyncio.run(api.global_search(   # global = map-reduce over reports
        config=_config(),
        entities=_read("entities"), communities=_read("communities"),
        community_reports=_read("community_reports"),
        community_level=COMMUNITY_LEVEL,      # 2 = mid-depth in the community hierarchy
        dynamic_community_selection=False, query=query))
    return GraphResult(str(response), _doc_ids_from_context(context), context)
```

```python
def graph_local(query, ...):
    response, context = asyncio.run(api.local_search(    # local = vector-search + one-hop
        config=_config(),
        entities=..., communities=..., community_reports=...,
        text_units=_read("text_units"), relationships=_read("relationships"),
        covariates=None,                      # claims extraction disabled in settings.yaml
        community_level=COMMUNITY_LEVEL, query=query))
    return GraphResult(str(response), _doc_ids_from_context(context), context)
```
Key contrasts to point out:
- **global** needs only `entities/communities/community_reports` (it reads summaries).
- **local** also needs `text_units` + `relationships` (the one-hop neighborhood).
- `_doc_ids_from_context()` is **best-effort** — it regex-scrapes `doc_NNNN` from the
  context; GraphRAG uses its own ids, so this often comes back empty (known limitation).

---

## 7. `harness/ab.py` — fan one query to all three routes

The heart of the showcase: same question, three architectures, isolated failures.

```python
_ROUTES = {"plain": _plain, "graphrag_global": _global, "graphrag_local": _local}
```
Each `_plain/_global/_local` is a thin adapter returning a uniform `RouteOutput`:
```python
def _plain(query):
    r = PlainRAG().answer(query)
    return RouteOutput("plain", r.response, r.retrieved_doc_ids)
```

```python
def run_route(name, query):
    try:
        return _ROUTES[name](query)
    except Exception as e:                     # one broken route must NOT kill the others
        return RouteOutput(name, "", [], error=f"{type(e).__name__}: {e}")

def run_all(query, routes=None):
    return {name: asdict(run_route(name, query)) for name in (routes or _ROUTES)}
```
Imports are **lazy** (inside the adapters) so calling only `plain` never loads GraphRAG.

---

## 8. `eval/run_eval.py` — ground-truth scoring

Why it matters: the lecture's "build evaluation before optimization." Separates
**retrieval** metrics from **generation** metrics, sliced by `question_type`.

```python
def stratified_sample(qa, limit):              # equal-ish count per question_type
    by_type[q["question_type"]].append(q); ... return out[:limit]
```

```python
def recall_at_k(retrieved, gold, k):
    if not gold: return nan                     # null queries have no gold docs
    return len(set(retrieved[:k]) & set(gold)) / len(gold)

def mrr(retrieved, gold):                       # 1/rank of first correct doc
    for i, d in enumerate(retrieved, 1):
        if d in set(gold): return 1.0/i
    return 0.0
```

```python
def judge(client, question, gold, candidate):   # LLM-as-judge for answer correctness
    # system prompt: CORRECT/WRONG; if gold == "Insufficient information.",
    # candidate is CORRECT only if it ALSO abstains (the hallucination test)
    return resp...startswith("CORRECT")
```

```python
def run(routes, limit, k):
    for q in stratified_sample(load_qa(), limit):
        for route in routes:
            out = run_route(route, q["query"])   # reuse the harness
            acc[route][qtype].append({
                "recall_at_k": recall_at_k(out.retrieved_doc_ids, gold_docs, k),
                "mrr":         mrr(out.retrieved_doc_ids, gold_docs),
                "correct":     judge(client, q["query"], q["answer"], out.response)})
    summary = _summarize(acc, k)                 # mean per (route, question_type) + ALL
    _report(summary, k, len(qa))                 # prints the table
    write(RESULTS_DIR / f"eval-{stamp}.json")    # saved for the slide
```
Run: `python -m eval.run_eval --limit 60 --routes plain,graphrag_local --k 8`.
Headline result to expect: plain RAG accuracy drops on `comparison`/`temporal`.

---

## 9. `app.py` — Gradio App

```python
EXAMPLES = [...]    
ROUTE_TITLES = {"plain": "🔎 Plain top-k RAG (baseline)", ...}
```

```python
def run_all_ui(query):
    for route in ("plain","graphrag_global","graphrag_local"):
        out = run_route(query, route)            # same harness the eval uses
        results.extend([out.response, _format_trace(out)])   # answer + retrieval trace
    return results
```

```python
with gr.Blocks(...) as demo:
    query = gr.Textbox(...); gr.Examples(EXAMPLES, ...)
    run_btn = gr.Button("Run all three")
    with gr.Row():                               # 3 columns side by side
        for route in (...):
            ans = gr.Markdown(); 
            with gr.Accordion("Retrieval trace", open=False): trace = gr.Markdown()
    run_btn.click(run_all_ui, inputs=query, outputs=outputs)   # wire button → fn → panels
```
`_format_trace()` shows the retrieved doc ids, or the per-route error if a route
failed (e.g. index not built). Launch: `python app.py` → http://127.0.0.1:7860.

---

## 10. One-line summary per file

| File | Role |
|---|---|
| `common.py` | shared models/paths/keys + `embed_texts` (single source of truth) |
| `ingest/subset.py` | 609 → 100-article tech subset + intact QA + `input/*.txt` |
| `settings.yaml` | GraphRAG config: models, news entity types |
| `baseline/plain_rag.py` | the control: chunk → embed → Chroma → top-k answer |
| `graphrag_app/query.py` | wrap GraphRAG global (reports) + local (one-hop) |
| `harness/ab.py` | one query → all 3 routes, failure-isolated |
| `eval/run_eval.py` | Recall@K, MRR, accuracy by question_type |
| `app.py` | Gradio 3-panel side-by-side UI |
