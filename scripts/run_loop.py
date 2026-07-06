"""Autonomous research loop — the bandit scheduler closes the plan→run→learn cycle.

Each iteration: Thompson scheduler picks explore-vs-mutate → planner/mutator
generates a config → run_single_experiment scores it → the scheduler's posterior
is updated with the reward. All state (experiments + bandit posteriors) persists
per-iteration, so ctrl-C loses nothing.

Usage:
    python scripts/run_loop.py --iterations 10
    python scripts/run_loop.py --iterations 5 --sleep 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.data_loader import DataLoader
from core.experiment import run_single_experiment
from core.memory import ExperimentStore
from core.memory_analyzer import effective_score
from core.mutator import generate_mutation
from core.planner import plan_next_research, suggestion_to_config
from core.scheduler import EXPLORE_ARM, ThompsonScheduler


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:30] or "direction"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the autonomous research loop")
    parser.add_argument("--iterations", type=int, default=10, help="Number of experiments to run")
    parser.add_argument("--db", default="db/experiments.db", help="SQLite database path")
    parser.add_argument("--no-cache", action="store_true", help="Ignore parquet cache")
    parser.add_argument(
        "--max-consecutive-failures", type=int, default=3,
        help="Stop after this many consecutive non-completed iterations",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0,
        help="Seconds to sleep between iterations (API rate limits)",
    )
    args = parser.parse_args()

    store = ExperimentStore(db_path=args.db)
    loader = DataLoader(no_cache=args.no_cache)
    scheduler = ThompsonScheduler(store)

    # One batch_id for the whole session → one ring in the graph visualization.
    batch_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_id = f"loop_{batch_ts}"

    exp_dir = Path("experiments")
    exp_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("AlphaResearchBot — Autonomous Research Loop")
    print(f"{'='*60}")
    print(f"Batch: {batch_id} | Iterations: {args.iterations} | DB: {args.db}\n")

    consecutive_failures = 0
    results: list[str] = []

    try:
        for i in range(1, args.iterations + 1):
            action, parent_id = scheduler.select_action()
            arm_id = EXPLORE_ARM if action == "explore" else f"mutate:{parent_id}"
            print(f"\n{'─'*60}")
            print(f"[{i}/{args.iterations}] Scheduler action: {arm_id}")
            print(f"{'─'*60}")

            # ── Generate config ───────────────────────────────────────────────
            parent_record = None
            try:
                if action == "explore":
                    suggestions = plan_next_research(store, n=1)
                    if not suggestions:
                        raise RuntimeError("planner produced no suggestions")
                    s = suggestions[0]
                    alpha_id = f"loop_{batch_ts}_{i:03d}_{_slug(s['direction'])}"
                    all_records = store.load_all()
                    base = all_records[0]["config"] if all_records else {}
                    config = suggestion_to_config(s, alpha_id, batch_id, base)
                else:
                    parent_record = store.load_by_id(parent_id)
                    config = generate_mutation(parent_id, store)
                    config["batch_id"] = batch_id
            except Exception as exc:
                print(f"  Config generation failed: {exc}")
                scheduler.update(arm_id, 0.0)
                consecutive_failures += 1
                results.append(f"[{i}] {arm_id} → generation failed | reward 0.00")
                if consecutive_failures >= args.max_consecutive_failures:
                    print(f"\nStopping: {consecutive_failures} consecutive failures.")
                    break
                continue

            # Persist the config JSON for reproducibility (same as manual scripts)
            config_path = exp_dir / f"{config['alpha_id']}.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            # ── Run experiment ────────────────────────────────────────────────
            outcome = run_single_experiment(
                config, store, loader=loader, force=False, verbose=True
            )

            # ── Reward + posterior update (persisted immediately) ────────────
            reward = scheduler.reward_for(action, parent_record, outcome)
            scheduler.update(arm_id, reward)

            if outcome.status == "completed" and outcome.record is not None:
                consecutive_failures = 0
                child_score = effective_score(outcome.record)
                parent_note = (
                    f" (parent {effective_score(parent_record):.1f})"
                    if parent_record else ""
                )
                line = (
                    f"[{i}] {arm_id} → {config['alpha_id']} | "
                    f"score {child_score:.1f}{parent_note} | "
                    f"reward {reward:.2f} | {outcome.record['verdict']}"
                )
            else:
                consecutive_failures += 1
                line = f"[{i}] {arm_id} → {outcome.status} | reward {reward:.2f}"
            print(f"\n  {line}")
            results.append(line)

            if consecutive_failures >= args.max_consecutive_failures:
                print(f"\nStopping: {consecutive_failures} consecutive failures.")
                break

            if args.sleep > 0 and i < args.iterations:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\n\nInterrupted — state is persisted per-iteration, nothing lost.")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Loop summary")
    print(f"{'='*60}")
    for line in results:
        print(f"  {line}")
    print("\nBandit posteriors:")
    for line in scheduler.summary_lines():
        print(line)

    records = store.load_all()
    if records:
        best = max(records, key=effective_score)
        print(
            f"\nBest alpha overall: {best['alpha_id']} "
            f"(score {effective_score(best):.1f}, verdict {best.get('verdict')})"
        )
    print(f"\nRun `python scripts/export_graph.py` to visualize the research graph.\n")


if __name__ == "__main__":
    main()
