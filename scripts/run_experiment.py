"""Run a single alpha experiment end-to-end (thin CLI wrapper around core.experiment).

Usage:
    python scripts/run_experiment.py --config experiments/sample_alpha_001.json
    python scripts/run_experiment.py --config experiments/sample_alpha_001.json --no-cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.data_loader import DataLoader
from core.experiment import run_single_experiment
from core.memory import ExperimentStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an alpha experiment")
    parser.add_argument("--config", required=True, help="Path to alpha JSON config")
    parser.add_argument("--db", default="db/experiments.db", help="SQLite database path")
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore parquet cache and force fresh data fetch"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Run even if the alpha is a near-duplicate of a prior one",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        alpha = json.load(f)

    print(f"\n{'='*60}")
    print(f"AlphaResearchBot — Experiment: {alpha.get('alpha_id', 'unknown')}")
    print(f"{'='*60}\n")
    print(f"Hypothesis : {alpha.get('hypothesis', '')}")
    print(f"Formula    : {alpha.get('formula', '')}")
    print(f"Universe   : {alpha.get('universe', '')}  |  "
          f"Rebalance: {alpha.get('rebalance', '')}  |  "
          f"Neutralization: {alpha.get('neutralization', '')}")
    print()

    store = ExperimentStore(db_path=args.db)
    loader = DataLoader(no_cache=args.no_cache)

    outcome = run_single_experiment(
        alpha, store, loader=loader, force=args.force, verbose=True
    )

    if outcome.status != "completed":
        print(f"Experiment did not complete: {outcome.status} — {outcome.error}")
        sys.exit(1)

    print(f"{'='*60}")
    print("Done. Run `python scripts/export_graph.py` to visualize the research graph.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
