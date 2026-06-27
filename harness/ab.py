"""A/B harness: fan one question out to all three routes and collect results.

This is the heart of the showcase -- same question, three architectures, side by
side. Each route is isolated in try/except so a not-yet-built index on one side
doesn't break the others.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RouteOutput:
    name: str
    response: str
    retrieved_doc_ids: list[str]
    error: str | None = None


def _plain(query: str) -> RouteOutput:
    from baseline.plain_rag import PlainRAG

    r = PlainRAG().answer(query)
    return RouteOutput("plain", r.response, r.retrieved_doc_ids)


def _global(query: str) -> RouteOutput:
    from graphrag_app.query import graph_global

    r = graph_global(query)
    return RouteOutput("graphrag_global", r.response, r.retrieved_doc_ids)


def _local(query: str) -> RouteOutput:
    from graphrag_app.query import graph_local

    r = graph_local(query)
    return RouteOutput("graphrag_local", r.response, r.retrieved_doc_ids)


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
