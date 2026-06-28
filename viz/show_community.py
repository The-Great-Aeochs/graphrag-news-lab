"""Render one GraphRAG community as an interactive graph + print its stored summary.

This makes the abstract "Leiden community + community report" concrete: you see
the entity nodes and relationship edges that the LLM read, next to the natural-
language report it wrote about them at index time.

Where things live (read straight from the index artifacts):
  output/communities.parquet       -> membership: which entities/relationships are in a community
  output/community_reports.parquet -> the LLM-written summary of that community
  output/entities.parquet          -> node titles, types, degree
  output/relationships.parquet     -> edges (source title -> target title) + descriptions

Usage:
    python viz/show_community.py --search FTX        # pick best community matching a keyword
    python viz/show_community.py --community 272      # pick a specific community id
    python viz/show_community.py --search AI --level 2
"""

from __future__ import annotations

import argparse
import textwrap
import webbrowser
from pathlib import Path

import pandas as pd
from pyvis.network import Network

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
VIZ = ROOT / "viz"

# One stable colour per entity type so the picture reads at a glance.
TYPE_COLORS = {
    "organization": "#4e79a7",
    "person": "#e15759",
    "product": "#59a14f",
    "technology": "#f28e2b",
    "event": "#b07aa1",
    "location": "#76b7b2",
    "financial_concept": "#edc948",
}


def pick_community(reports: pd.DataFrame, search: str | None, community: int | None,
                   level: int | None) -> pd.Series:
    df = reports
    if level is not None:
        df = df[df["level"] == level]
    if community is not None:
        return df[df["community"] == community].iloc[0]
    if search:
        hits = df[df["title"].str.contains(search, case=False, na=False)]
        if hits.empty:
            raise SystemExit(f"No community title matches '{search}'.")
        # most legible: prefer richer-but-not-huge communities, ranked by importance
        return hits.sort_values(["rank", "size"], ascending=False).iloc[0]
    # default: the highest-ranked mid-size community
    return df.sort_values("rank", ascending=False).iloc[0]


def print_summary(rep: pd.Series) -> None:
    bar = "=" * 78
    print(f"\n{bar}\nCOMMUNITY {rep['community']}  (level {rep['level']}, "
          f"{rep['size']} entities, rank {rep['rank']})\n{bar}")
    print(f"TITLE: {rep['title']}\n")
    print("SUMMARY (stored in output/community_reports.parquet -> 'summary'):")
    print(textwrap.fill(str(rep["summary"]), 78), "\n")
    findings = rep.get("findings")
    if findings is not None and hasattr(findings, "__len__") and len(findings):
        print("KEY FINDINGS (-> 'findings'):")
        for f in list(findings)[:5]:
            summary = f.get("summary") if isinstance(f, dict) else str(f)
            print("  • " + textwrap.fill(str(summary), 74, subsequent_indent="    "))
    print(bar)


def build_graph(rep: pd.Series) -> Path:
    comm = pd.read_parquet(OUT / "communities.parquet")
    ents = pd.read_parquet(OUT / "entities.parquet")
    rels = pd.read_parquet(OUT / "relationships.parquet")

    row = comm[comm["community"] == rep["community"]].iloc[0]
    ent_ids = set(row["entity_ids"])
    rel_ids = set(row["relationship_ids"])

    nodes = ents[ents["id"].isin(ent_ids)]
    edges = rels[rels["id"].isin(rel_ids)]
    titles = set(nodes["title"])

    net = Network(height="800px", width="100%", bgcolor="#111", font_color="#eee",
                  notebook=False, directed=False)
    net.barnes_hut(gravity=-8000, spring_length=120)

    for _, e in nodes.iterrows():
        color = TYPE_COLORS.get(str(e["type"]).lower(), "#999")
        size = 12 + float(e.get("degree") or 0) * 1.5     # bigger = more connected
        tooltip = f"<b>{e['title']}</b> ({e['type']})<br>{str(e['description'])[:300]}"
        net.add_node(e["title"], label=e["title"], color=color, size=size, title=tooltip)

    for _, r in edges.iterrows():
        if r["source"] in titles and r["target"] in titles:   # keep edges inside the community
            net.add_edge(r["source"], r["target"], value=float(r.get("weight") or 1),
                         title=str(r["description"])[:300])

    VIZ.mkdir(exist_ok=True)
    out = VIZ / f"community_{rep['community']}.html"
    net.write_html(str(out), notebook=False)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--search", default="FTX", help="keyword to match a community title")
    p.add_argument("--community", type=int, default=None, help="exact community id")
    p.add_argument("--level", type=int, default=None, help="restrict to a hierarchy level")
    p.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    args = p.parse_args()

    reports = pd.read_parquet(OUT / "community_reports.parquet")
    rep = pick_community(reports, args.search, args.community, args.level)
    print_summary(rep)
    html = build_graph(rep)
    print(f"\ngraph written -> {html}")
    if not args.no_open:
        webbrowser.open(f"file://{html}")
