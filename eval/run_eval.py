"""Ground-truth eval: plain RAG vs GraphRAG, sliced by question_type.

The MultiHop-RAG QA pairs give us, for free:
  - evidence_doc_ids  -> retrieval metrics (Recall@K, MRR)
  - answer            -> generation correctness (via an LLM judge)
  - question_type     -> slice results by reasoning type (the headline: plain RAG
                         collapses on comparison/temporal/multi-hop, GraphRAG holds)
  - null_query        -> abstention test ("fail gracefully, don't hallucinate")

This separates retrieval metrics from generation metrics on purpose -- the
lecture's rule: you must be able to tell a retrieval failure from a generation
failure.

Usage:
    python -m eval.run_eval --limit 60 --routes plain,graphrag_local
    python -m eval.run_eval --limit 120 --routes plain,graphrag_local --k 8
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from common import CHAT_MODEL, ROOT, get_openai_client, load_qa

RESULTS_DIR = ROOT / "eval" / "results"
JUDGE_SYSTEM = (
    "You are a strict grader for a question-answering system. You are given a "
    "QUESTION, the GOLD answer, and a CANDIDATE answer. Reply with a single word: "
    "CORRECT or WRONG.\n"
    "- If the GOLD answer is 'Insufficient information.', the CANDIDATE is CORRECT "
    "only if it declines to answer or says it lacks enough information; otherwise "
    "WRONG (it hallucinated).\n"
    "- Otherwise CANDIDATE is CORRECT if it conveys the same factual answer as "
    "GOLD, even if phrased differently or with extra detail."
)


def stratified_sample(qa: list[dict], limit: int, seed: int = 7) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in qa:
        by_type[q["question_type"]].append(q)
    rng = random.Random(seed)
    per = max(1, limit // len(by_type))
    out: list[dict] = []
    for t, items in by_type.items():
        rng.shuffle(items)
        out.extend(items[:per])
    rng.shuffle(out)
    return out[:limit]


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return float("nan")  # null queries have no gold docs
    topk = set(retrieved[:k])
    return len(topk & set(gold)) / len(gold)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    goldset = set(gold)
    for i, d in enumerate(retrieved, 1):
        if d in goldset:
            return 1.0 / i
    return 0.0


def judge(client, question: str, gold: str, candidate: str) -> bool:
    msg = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": f"QUESTION: {question}\nGOLD: {gold}\nCANDIDATE: {candidate}",
        },
    ]
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=msg, temperature=0.0)
    return resp.choices[0].message.content.strip().upper().startswith("CORRECT")


def run(routes: list[str], limit: int, k: int) -> dict:
    from harness.ab import run_route

    qa = stratified_sample(load_qa(), limit)
    client = get_openai_client()
    # accumulator: route -> qtype -> list of per-question records
    acc: dict = {r: defaultdict(list) for r in routes}

    for i, q in enumerate(qa, 1):
        qtype = q["question_type"]
        gold_docs = q.get("evidence_doc_ids", [])
        print(f"[{i}/{len(qa)}] ({qtype}) {q['query'][:70]}...")
        for route in routes:
            out = run_route(route, q["query"])
            if out.error:
                print(f"    {route}: ERROR {out.error}")
                continue
            rec = {
                "recall_at_k": recall_at_k(out.retrieved_doc_ids, gold_docs, k),
                "mrr": mrr(out.retrieved_doc_ids, gold_docs),
                "correct": judge(client, q["query"], q["answer"], out.response),
            }
            acc[route][qtype].append(rec)

    summary = _summarize(acc, k)
    _report(summary, k, len(qa))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"eval-{stamp}.json"
    out_path.write_text(json.dumps({"k": k, "n": len(qa), "summary": summary}, indent=2))
    print(f"\nsaved -> {out_path}")
    return summary


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def _summarize(acc: dict, k: int) -> dict:
    summary: dict = {}
    for route, by_type in acc.items():
        summary[route] = {}
        all_recall, all_mrr, all_correct = [], [], []
        for qtype, recs in by_type.items():
            recalls = [r["recall_at_k"] for r in recs]
            mrrs = [r["mrr"] for r in recs]
            corrects = [1.0 if r["correct"] else 0.0 for r in recs]
            summary[route][qtype] = {
                "n": len(recs),
                f"recall@{k}": round(_mean(recalls), 3),
                "mrr": round(_mean(mrrs), 3),
                "accuracy": round(_mean(corrects), 3),
            }
            all_recall += recalls
            all_mrr += mrrs
            all_correct += corrects
        summary[route]["ALL"] = {
            "n": len(all_correct),
            f"recall@{k}": round(_mean(all_recall), 3),
            "mrr": round(_mean(all_mrr), 3),
            "accuracy": round(_mean(all_correct), 3),
        }
    return summary


def _report(summary: dict, k: int, n: int) -> None:
    print("\n" + "=" * 72)
    print(f"EVAL SUMMARY  (n={n}, k={k})")
    print("=" * 72)
    for route, by_type in summary.items():
        print(f"\n### {route}")
        print(f"{'question_type':<20}{'n':>5}{'recall@'+str(k):>12}{'mrr':>8}{'accuracy':>10}")
        for qtype, m in by_type.items():
            print(
                f"{qtype:<20}{m['n']:>5}{m[f'recall@{k}']:>12}{m['mrr']:>8}{m['accuracy']:>10}"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--routes", default="plain,graphrag_local")
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--k", type=int, default=8)
    args = p.parse_args()
    run(args.routes.split(","), args.limit, args.k)
