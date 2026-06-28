"""A/B harness: fan one question out to all three routes and collect results.

This is the heart of the showcase -- same question, three architectures, side by
side. Each route is isolated in try/except so a not-yet-built index on one side
doesn't break the others.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass
class RouteOutput:
    name: str
    response: str
    retrieved_doc_ids: list[str]
    error: str | None = None
    trace: str = ""  # human-readable, route-native retrieval trace (markdown)


def _plain(query: str) -> RouteOutput:
    from baseline.plain_rag import PlainRAG

    r = PlainRAG().answer(query)
    seen, lines = set(), []
    for c in r.retrieved_chunks:  # what plain RAG actually retrieves: text chunks
        if c["title"] not in seen:
            seen.add(c["title"])
            lines.append(f"- {c['title']} — *{c['source']}*")
    trace = f"**📄 Retrieved chunks** (top-{len(seen)} by cosine):\n" + "\n".join(lines)
    return RouteOutput("plain", r.response, r.retrieved_doc_ids, trace=trace)


def _global(query: str) -> RouteOutput:
    from graphrag_app.query import graph_global

    r = graph_global(query)
    n_considered = len(r.trace.get("reports", []))
    # The reports the answer actually drew on are cited inline as [Data: Reports (2, 6, ..)].
    by_id = r.trace.get("reports_by_id", {})
    groups = re.findall(r"Reports?\s*\(([^)]*)\)", r.response)
    cited_ids = set(re.findall(r"\d+", " ".join(groups)))
    used = sorted({by_id[i] for i in cited_ids if i in by_id})
    if used:
        body = "\n".join(f"  - {x}" for x in used)
        detail = f"**Community reports the answer cited ({len(used)})**:\n{body}"
    else:
        body = "\n".join(f"  - {x}" for x in r.trace.get("reports", [])[:10])
        detail = f"**Top community reports by centrality:**\n{body}"
    trace = (
        f"**🌐 Map-reduce over {n_considered} community reports** — global search reads "
        f"*summaries*, not documents.\n\n{detail}"
    )
    return RouteOutput("graphrag_global", r.response, r.retrieved_doc_ids, trace=trace)


def _local(query: str) -> RouteOutput:
    from graphrag_app.query import graph_local

    r = graph_local(query)
    t = r.trace
    ents = t.get("entities", [])
    rels = t.get("relationships", [])
    reps = t.get("reports", [])
    rel_sample = "; ".join(f"{s} → {tg}" for s, tg in rels[:6])
    rep_lines = "\n".join(f"  - {x}" for x in reps)
    trace = (
        f"**🔵 Entities in one-hop neighbourhood ({len(ents)})**: {', '.join(ents[:12])}"
        f"{' …' if len(ents) > 12 else ''}\n\n"
        f"**↔ Relationships traversed ({len(rels)})**: {rel_sample}"
        f"{' …' if len(rels) > 6 else ''}\n\n"
        f"**📄 Community reports ({len(reps)})**:\n{rep_lines}\n\n"
        f"**📑 Source passages pulled**: {t.get('sources', 0)}"
    )
    return RouteOutput("graphrag_local", r.response, r.retrieved_doc_ids, trace=trace)


_ROUTES = {"plain": _plain, "graphrag_global": _global, "graphrag_local": _local}


def run_route(name: str, query: str) -> RouteOutput:
    try:
        return _ROUTES[name](query)
    except Exception as e:  # noqa: BLE001 -- surface failures per route, don't crash
        return RouteOutput(name, "", [], error=f"{type(e).__name__}: {e}")


def run_all(query: str, routes: list[str] | None = None) -> dict[str, dict]:
    routes = routes or list(_ROUTES)
    return {name: asdict(run_route(name, query)) for name in routes}


if __name__ == "__main__":
    import json
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "What are the main themes across this corpus?"
    print(json.dumps(run_all(q), indent=2))
