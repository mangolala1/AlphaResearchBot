"""Export the research graph to an interactive HTML file.

Usage:
    python scripts/export_graph.py
    python scripts/export_graph.py --db db/experiments.db --output reports/research_graph.html
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
    args = parser.parse_args()

    store = ExperimentStore(db_path=args.db)
    records = store.load_all()

    if not records:
        print("No experiments found in the database. Run run_experiment.py first.")
        sys.exit(0)

    research_graph = ResearchGraph()
    research_graph.build_from_store(store)
    g = research_graph.get_graph()

    export_graph_html(g, output_path=args.output)

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    print(f"Graph exported to {args.output} ({n_nodes} node(s), {n_edges} edge(s))")


if __name__ == "__main__":
    main()
