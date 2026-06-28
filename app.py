"""Gradio showcase: one question, three architectures, side by side.

Left to right: plain top-k RAG (baseline) | GraphRAG global | GraphRAG local.
Each panel shows the answer plus an expandable "what got retrieved" trace --
the lecture's "make retrieval observable" rule, made visual. The contrast is the
point: ask a sensemaking question and watch plain RAG flail while global search
answers it.

Run:
    python app.py
"""

from __future__ import annotations

import gradio as gr

from harness.ab import run_route

EXAMPLES = [
    # (label, query) -- chosen to exercise each pitfall
    "What are the main themes and storylines across this news corpus?",  # global / sensemaking
    "What are the biggest narratives in AI this period and who are the key players?",  # global
    "Who is Sam Bankman-Fried and what is he connected to?",  # local / entity-centric
    "How is OpenAI connected to other companies and people in the news?",  # local
    "Which companies are mentioned alongside both layoffs and AI investments?",  # multi-hop-ish
    "What did the CEO of Acme Robotics say about quarterly earnings?",  # null / should abstain
]

ROUTE_TITLES = {
    "plain": "🔎 Plain top-k RAG (baseline)",
    "graphrag_global": "🌐 GraphRAG — Global search",
    "graphrag_local": "📍 GraphRAG — Local search",
}


def _format_trace(out) -> str:
    if out.error:
        return f"**⚠️ {out.error}**\n\n(Has the index been built? See README.)"
    return out.trace or "_(no trace)_"


def run_one(query: str, route: str):
    out = run_route(route, query)
    answer = out.response or "_(no response)_"
    return answer, _format_trace(out)


def run_all_ui(query: str):
    results = []
    for route in ("plain", "graphrag_global", "graphrag_local"):
        ans, trace = run_one(query, route)
        results.extend([ans, trace])
    return results


with gr.Blocks(title="GraphRAG News Lab") as demo:
    gr.Markdown(
        "# GraphRAG News Lab\n"
        "One question → **three architectures**, side by side, on the MultiHop-RAG "
        "news corpus. Plain top-k RAG is the control; GraphRAG global handles "
        "*sensemaking* (themes across the corpus), GraphRAG local handles "
        "*entity-centric* questions (one-hop neighbourhood). Watch where the "
        "baseline breaks."
    )

    query = gr.Textbox(
        label="Ask the corpus",
        placeholder="e.g. What are the main themes across this news corpus?",
        lines=2,
    )
    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=query, label="Try these")
    run_btn = gr.Button("Run all three", variant="primary")

    panels = {}
    with gr.Row():
        for route in ("plain", "graphrag_global", "graphrag_local"):
            with gr.Column():
                gr.Markdown(f"### {ROUTE_TITLES[route]}")
                ans = gr.Markdown(label="Answer")
                with gr.Accordion("Retrieval trace", open=False):
                    trace = gr.Markdown()
                panels[route] = (ans, trace)

    outputs = []
    for route in ("plain", "graphrag_global", "graphrag_local"):
        outputs.extend(panels[route])

    run_btn.click(run_all_ui, inputs=query, outputs=outputs)
    query.submit(run_all_ui, inputs=query, outputs=outputs)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
