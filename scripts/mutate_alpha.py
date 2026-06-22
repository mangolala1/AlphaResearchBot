"""Generate a mutated child alpha from a parent experiment.

Usage:
    python scripts/mutate_alpha.py --parent alpha_001
    python scripts/mutate_alpha.py --parent alpha_001 --run
    python scripts/mutate_alpha.py --parent alpha_001 --run --db db/experiments.db
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.memory import ExperimentStore
from core.mutator import generate_mutation


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a mutated child alpha from a parent")
    parser.add_argument("--parent", required=True, help="alpha_id of the parent experiment")
    parser.add_argument("--db", default="db/experiments.db", help="SQLite database path")
    parser.add_argument("--run", action="store_true", help="Run the generated alpha immediately after saving")
    args = parser.parse_args()

    store = ExperimentStore(db_path=args.db)

    print(f"\nGenerating mutation from parent: {args.parent}\n")
    try:
        child = generate_mutation(args.parent, store)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # Save to experiments/
    out_path = Path("experiments") / f"{child['alpha_id']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(child, f, indent=2)

    print("Generated child alpha:")
    print(json.dumps(child, indent=2))
    print(f"\nSaved to: {out_path}")

    if args.run:
        print(f"\nRunning experiment: {child['alpha_id']}\n")
        result = subprocess.run(
            [sys.executable, "scripts/run_experiment.py", "--config", str(out_path), "--db", args.db],
            check=False,
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
