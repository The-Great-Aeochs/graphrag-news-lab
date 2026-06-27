"""Build a ~100-article subset of MultiHop-RAG for the GraphRAG showcase.

Strategy
--------
The full corpus is 609 news articles across 6 categories. Indexing all of them
with an LLM (entity/relationship extraction + community reports) costs real
tokens, so we iterate on a coherent subset first.

We keep the **technology** category only: it carries the richest narrative for a
GraphRAG demo (AI labs, the FTX/crypto saga, Big Tech moves) and the densest
web of shared entities -- exactly what makes global sensemaking and entity-centric
local search shine.

To preserve as many *intact* multi-hop QA pairs as possible, we rank tech
articles by how often the QA evidence references them and keep the top N. A QA
pair is kept only if **every** article in its evidence_list survives in the
subset -- otherwise its ground-truth answer is unreachable and it would poison
the eval.

We also keep a handful of `null_query` pairs (questions the corpus cannot
answer) to demonstrate graceful failure -- the lecture's "fail, don't
hallucinate" rule.

Outputs
-------
data/subset/corpus_subset.json   the kept articles
data/subset/qa_subset.json       the kept QA pairs (intact evidence + nulls)
input/*.txt                      one article per file, for `graphrag index`
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SUBSET = ROOT / "data" / "subset"
INPUT = ROOT / "input"

CATEGORY = "technology"
TARGET_ARTICLES = 100
N_NULL_QUERIES = 15


def slugify(text: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return s[:maxlen] or "untitled"


def main() -> None:
    corpus = json.loads((RAW / "corpus.json").read_text())
    qa = json.loads((RAW / "MultiHopRAG.json").read_text())

    # Stable id per article (index into original corpus) + url lookup.
    for i, art in enumerate(corpus):
        art["doc_id"] = f"doc_{i:04d}"
    url2art = {a["url"]: a for a in corpus}

    tech = [a for a in corpus if a["category"] == CATEGORY]

    # Rank tech articles by how many QA evidence items reference them.
    ref_counts: Counter = Counter()
    for q in qa:
        for ev in q["evidence_list"]:
            art = url2art.get(ev["url"])
            if art and art["category"] == CATEGORY:
                ref_counts[art["url"]] += 1

    tech.sort(key=lambda a: ref_counts[a["url"]], reverse=True)
    kept = tech[:TARGET_ARTICLES]
    kept_urls = {a["url"] for a in kept}

    # Keep QA pairs whose evidence is fully inside the subset.
    intact = [
        q
        for q in qa
        if q["question_type"] != "null_query"
        and q["evidence_list"]
        and all(ev["url"] in kept_urls for ev in q["evidence_list"])
    ]
    nulls = [q for q in qa if q["question_type"] == "null_query"][:N_NULL_QUERIES]
    qa_subset = intact + nulls

    # Attach resolved doc_ids to each QA pair's evidence for easy eval scoring.
    for q in qa_subset:
        q["evidence_doc_ids"] = [
            url2art[ev["url"]]["doc_id"]
            for ev in q["evidence_list"]
            if ev["url"] in url2art
        ]

    SUBSET.mkdir(parents=True, exist_ok=True)
    (SUBSET / "corpus_subset.json").write_text(json.dumps(kept, indent=2))
    (SUBSET / "qa_subset.json").write_text(json.dumps(qa_subset, indent=2))

    # Write one .txt per article for the GraphRAG indexer. We prepend title +
    # source + date so the extractor sees publication context (helps temporal
    # reasoning and entity grounding).
    if INPUT.exists():
        for f in INPUT.glob("*.txt"):
            f.unlink()
    INPUT.mkdir(parents=True, exist_ok=True)
    for art in kept:
        header = (
            f"Title: {art['title']}\n"
            f"Source: {art['source']}\n"
            f"Published: {art['published_at']}\n"
            f"Category: {art['category']}\n\n"
        )
        fname = f"{art['doc_id']}_{slugify(art['title'])}.txt"
        (INPUT / fname).write_text(header + (art.get("body") or ""))

    print(f"kept articles      : {len(kept)}")
    print(f"intact multi-hop QA: {len(intact)}")
    print(f"  by type          : {dict(Counter(q['question_type'] for q in intact))}")
    print(f"null queries kept  : {len(nulls)}")
    print(f"total QA in subset : {len(qa_subset)}")
    print(f"input .txt files   : {len(list(INPUT.glob('*.txt')))}")


if __name__ == "__main__":
    main()
