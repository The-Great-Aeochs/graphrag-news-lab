"""Thin wrappers around GraphRAG's global and local search.

These call the official `graphrag.api` functions against the parquet artifacts
produced by `graphrag index`. Both return a normalized GraphResult so the A/B
harness can treat all three routes (global / local / plain) uniformly.

- global_search  -> Pitfall 1, sensemaking: map-reduce over community reports.
- local_search   -> Pitfall 2, entity-centric: vector-entry + one-hop expansion.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from common import GRAPHRAG_OUTPUT, ROOT

# Default community level for query time. Level 0 = a few broad communities,
# higher = tighter sub-communities. 2 is a reasonable mid-depth default.
COMMUNITY_LEVEL = 2
_DOC_ID_RE = re.compile(r"doc_\d{4}")


@dataclass
class GraphResult:
    response: str
    retrieved_doc_ids: list[str]
    context: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)


def _extract_trace(context: dict) -> dict:
    """Pull each route's *native* retrieval units out of the context object.

    GraphRAG doesn't retrieve documents -- it returns entities, relationships,
    community reports, and source text units. This surfaces them so the UI can
    show what the route actually used, instead of an empty doc-id list."""
    t: dict = {}
    df = context.get("entities")
    if isinstance(df, pd.DataFrame) and len(df):
        col = "entity" if "entity" in df.columns else "title"
        t["entities"] = [str(x) for x in df[col].tolist()]
    df = context.get("relationships")
    if isinstance(df, pd.DataFrame) and len(df):
        t["relationships"] = [(str(r["source"]), str(r["target"])) for _, r in df.iterrows()]
    df = context.get("reports")
    if isinstance(df, pd.DataFrame) and len(df):
        sort_col = next((c for c in ("occurrence weight", "rank") if c in df.columns), None)
        d = df.sort_values(sort_col, ascending=False) if sort_col else df
        t["reports"] = [str(x) for x in d["title"].tolist()]
        if "id" in df.columns:  # map id -> title so we can resolve inline [Data: Reports (..)]
            t["reports_by_id"] = {str(i): str(tt) for i, tt in zip(df["id"], df["title"])}
    df = context.get("sources")
    if isinstance(df, pd.DataFrame):
        t["sources"] = len(df)
    return t


@lru_cache(maxsize=1)
def _config():
    from graphrag.config.load_config import load_config

    return load_config(ROOT)


def _read(name: str) -> pd.DataFrame:
    path = GRAPHRAG_OUTPUT / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `graphrag index --root .` first "
            "(see scripts/run_index.sh)."
        )
    return pd.read_parquet(path)


def _doc_ids_from_context(context: dict) -> list[str]:
    """Best-effort: pull doc_NNNN ids out of whatever source records the search
    used. Our input files are named doc_NNNN_slug.txt, so the id survives in the
    text-unit / source titles. Used for GraphRAG-side Recall@K."""
    found, seen = [], set()
    for key in ("sources", "text_units", "reports", "entities", "relationships"):
        val = context.get(key)
        if isinstance(val, pd.DataFrame):
            blob = " ".join(val.astype(str).fillna("").values.ravel().tolist())
        elif isinstance(val, list):
            blob = " ".join(map(str, val))
        else:
            blob = str(val) if val is not None else ""
        for m in _DOC_ID_RE.findall(blob):
            if m not in seen:
                seen.add(m)
                found.append(m)
    return found


def graph_global(query: str, response_type: str = "multiple paragraphs") -> GraphResult:
    import graphrag.api as api

    response, context = asyncio.run(
        api.global_search(
            config=_config(),
            entities=_read("entities"),
            communities=_read("communities"),
            community_reports=_read("community_reports"),
            community_level=COMMUNITY_LEVEL,
            dynamic_community_selection=False,
            response_type=response_type,
            query=query,
        )
    )
    return GraphResult(str(response), _doc_ids_from_context(context), context,
                       _extract_trace(context))


def graph_local(query: str, response_type: str = "multiple paragraphs") -> GraphResult:
    import graphrag.api as api

    response, context = asyncio.run(
        api.local_search(
            config=_config(),
            entities=_read("entities"),
            communities=_read("communities"),
            community_reports=_read("community_reports"),
            text_units=_read("text_units"),
            relationships=_read("relationships"),
            covariates=None,  # claims extraction is disabled in settings.yaml
            community_level=COMMUNITY_LEVEL,
            response_type=response_type,
            query=query,
        )
    )
    return GraphResult(str(response), _doc_ids_from_context(context), context,
                       _extract_trace(context))
