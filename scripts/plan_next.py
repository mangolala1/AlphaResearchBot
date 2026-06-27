"""Research planner — reads experiment memory and suggests the next N alpha directions.

Usage:
    python scripts/plan_next.py
    python scripts/plan_next.py --n 5
    python scripts/plan_next.py --n 3 --save
    python scripts/plan_next.py --n 3 --save --db db/experiments.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.memory import ExperimentStore
from core.planner import plan_next_research


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest next alpha research directions")
    parser.add_argument("--n", type=int, default=3, help="Number of suggestions to generate (default: 3)")
    parser.add_argument("--db", default="db/experiments.db", help="SQLite database path")
    parser.add_argument("--save", action="store_true", help="Save each suggestion as a JSON file in experiments/")
    args = parser.parse_args()

    store = ExperimentStore(db_path=args.db)
    all_records = store.load_all()

    print(f"\n{'='*60}")
    print(f"AlphaResearchBot — Research Planner")
    print(f"{'='*60}")
    print(f"Experiments in memory: {len(all_records)}")
    print(f"Requesting {args.n} suggestions...\n")

    suggestions = plan_next_research(store, n=args.n)

    if not suggestions:
        print("No suggestions generated.")
        sys.exit(1)

    for i, s in enumerate(suggestions, 1):
        print(f"{'─'*60}")
        print(f"Suggestion {i}: {s['direction']}")
        print(f"  Hypothesis : {s['hypothesis']}")
        print(f"  Formula    : {s['formula']}")
        print(f"  Features   : {s['features']}")
        print(f"  Parent     : {s['parent_id'] or 'none (new branch)'}")
        print(f"  Rationale  : {s['rationale']}")
        print()

    if args.save:
        saved = []
        exp_dir = Path("experiments")
        exp_dir.mkdir(exist_ok=True)

        for i, s in enumerate(suggestions, 1):
            alpha_id = f"plan_{i:03d}_{s['direction'].lower().replace(' ', '_')}"
            # Use first experiment's config as template for required fields
            base = all_records[0]["config"] if all_records else {}
            config = {
                "alpha_id": alpha_id,
                "parent_id": s["parent_id"],
                "hypothesis": s["hypothesis"],
                "formula": s["formula"],
                "features": s["features"],
                "mutation": f"Planner suggestion: {s['direction']}",
                "universe": base.get("universe", "sp500"),
                "start_date": base.get("start_date", "2021-01-01"),
                "end_date": base.get("end_date", "2026-06-01"),
                "neutralization": base.get("neutralization", "sector"),
                "rebalance": base.get("rebalance", "monthly"),
                "transaction_cost_bps": base.get("transaction_cost_bps", 5),
                "holding_period_days": base.get("holding_period_days", 20),
            }
            out_path = exp_dir / f"{alpha_id}.json"
            with open(out_path, "w") as f:
                json.dump(config, f, indent=2)
            saved.append(str(out_path))

        print(f"{'='*60}")
        print(f"Saved {len(saved)} suggestion(s) to experiments/:")
        for p in saved:
            print(f"  {p}")
        print(f"\nRun any with:")
        print(f"  python scripts/run_experiment.py --config experiments/<file>.json")
        print()


if __name__ == "__main__":
    main()
