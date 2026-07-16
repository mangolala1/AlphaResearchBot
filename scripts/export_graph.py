"""Export the research graph to an interactive HTML file.

Usage:
    python scripts/export_graph.py
    python scripts/export_graph.py --db db/experiments.db --output reports/research_graph.html
    python scripts/export_graph.py --filter-verdict promising
    python scripts/export_graph.py --filter-top 10
    python scripts/export_graph.py --filter-since 2026-06-01
    python scripts/export_graph.py --filter-batch batch_20260627_120000
    python scripts/export_graph.py --filter-verdict promising --include-ancestors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.graph import ResearchGraph
from core.memory import ExperimentStore
from core.visualization import export_graph_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Export research graph to HTML")
    parser.add_argument("--db", default="db/experiments.db", help="SQLite database path")
    parser.add_argument(
        "--output", default="reports/research_graph.html", help="Output HTML file path"
    )
    parser.add_argument(
        "--filter-verdict", default=None,
        help="Comma-separated verdicts to keep (e.g. promising,revise)"
    )
    parser.add_argument(
        "--filter-top", type=int, default=None,
        help="Keep top N experiments by Sharpe"
    )
    parser.add_argument(
        "--filter-since", default=None,
        help="Keep experiments with timestamp >= DATE (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--filter-batch", default=None,
        help="Keep only experiments from a specific batch_id"
    )
    parser.add_argument(
        "--include-ancestors", action="store_true",
        help="Include all ancestor nodes of kept nodes via parent_id chain"
    )
    args = parser.parse_args()

    store = ExperimentStore(db_path=args.db)
    records = store.load_all()

    if not records:
        print("No experiments found in the database. Run run_experiment.py first.")
        sys.exit(0)

    research_graph = ResearchGraph()
    research_graph.build_from_store(store)
    g = research_graph.get_graph()

    # ── Filtering ────────────────────────────────────────────────────────────
    to_keep = set(g.nodes)

    if args.filter_verdict:
        verdicts = set(args.filter_verdict.split(","))
        to_keep &= {n for n in to_keep if g.nodes[n].get("verdict") in verdicts}

    if args.filter_since:
        to_keep &= {
            n for n in to_keep
            if g.nodes[n].get("timestamp", "") >= args.filter_since
        }

    if args.filter_batch:
        to_keep &= {
            n for n in to_keep
            if g.nodes[n].get("batch_id") == args.filter_batch
        }

    if args.filter_top is not None:
        # Rank by predictive magnitude when available (>= 0), matching parent-pool
        # semantics; directional score then Sharpe as fallbacks. Scored nodes
        # always outrank unscored pre-V4 nodes.
        def _rank_key(n):
            magnitude = g.nodes[n].get("predictive_magnitude", -1.0)
            if magnitude is not None and magnitude >= 0:
                return (1, magnitude)
            score = g.nodes[n].get("score", -1.0)
            if score is None or score < 0:
                return (0, g.nodes[n].get("Sharpe", 0.0))
            return (1, score)
        by_score = sorted(to_keep, key=_rank_key, reverse=True)
        to_keep = set(by_score[:args.filter_top])

    # ── Ancestor expansion ───────────────────────────────────────────────────
    if args.include_ancestors:
        for node in list(to_keep):
            parent = g.nodes[node].get("parent_id")
            while parent and parent in g.nodes:
                to_keep.add(parent)
                parent = g.nodes[parent].get("parent_id")

    g.remove_nodes_from(set(g.nodes) - to_keep)

    # ── Ring index from batch_id (sorted chronologically) ───────────────────
    batches = sorted(
        {g.nodes[n].get("batch_id") for n in g.nodes if g.nodes[n].get("batch_id")}
    )
    batch_order = {b: i for i, b in enumerate(batches)}
    for node in g.nodes:
        g.nodes[node]["ring"] = batch_order.get(g.nodes[node].get("batch_id"), 0)

    export_graph_html(g, output_path=args.output)

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    print(f"Graph exported to {args.output} ({n_nodes} node(s), {n_edges} edge(s))")
    if len(batches) > 1:
        print(f"  Rings: {len(batches)} batches → {batches}")


if __name__ == "__main__":
    main()
