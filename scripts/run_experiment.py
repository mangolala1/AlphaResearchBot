"""Run a single alpha experiment end-to-end.

Usage:
    python scripts/run_experiment.py --config experiments/sample_alpha_001.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.backtest import run_backtest
from core.decision import decide
from core.memory import ExperimentStore
from core.reflection import generate_reflection
from core.robustness import run_robustness
from core.types import ExperimentRecord
from core.validator import validate_alpha


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an alpha experiment")
    parser.add_argument("--config", required=True, help="Path to alpha JSON config")
    parser.add_argument("--db", default="db/experiments.db", help="SQLite database path")
    args = parser.parse_args()

    # Load alpha config
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

    # Validate
    print("[ Step 1 ] Validating formula...")
    result = validate_alpha(alpha)
    if result.warnings:
        for w in result.warnings:
            print(f"  WARNING: {w}")
    if not result.valid:
        print("\nValidation FAILED:")
        for e in result.errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    print("  Validation passed.\n")

    # Backtest
    print("[ Step 2 ] Running mock backtest...")
    metrics = run_backtest(alpha)
    print(f"  IC_mean        : {metrics['IC_mean']:.4f}")
    print(f"  ICIR           : {metrics['ICIR']:.4f}")
    print(f"  Sharpe         : {metrics['Sharpe']:.4f}")
    print(f"  Deflated Sharpe: {metrics['deflated_sharpe']:.4f}")
    print(f"  Turnover       : {metrics['turnover']:.1f} bps/yr")
    print(f"  Max Drawdown   : {metrics['max_drawdown']:.4f}")
    print(f"  Noise Risk     : {metrics['noise_risk']}\n")

    # Robustness
    print("[ Step 3 ] Running mock robustness checks...")
    robustness = run_robustness(alpha, metrics)
    print(f"  Sector Stability      : {robustness['sector_stability']:.4f}")
    print(f"  Subperiod Stability   : {robustness['subperiod_stability']:.4f}")
    print(f"  Market Regime Sharpe  : {robustness['market_regime_sharpe']:.4f}")
    print(f"  Placebo Score         : {robustness['placebo_score']:.4f}\n")

    # Decision
    print("[ Step 4 ] Applying decision logic...")
    verdict, failure_reason = decide(metrics, robustness)
    verdict_display = verdict.upper()
    print(f"  Verdict: {verdict_display}")
    if failure_reason:
        print(f"  Reason : {failure_reason}")
    print()

    # Reflection
    print("[ Step 5 ] Generating reflection...")
    reflection = generate_reflection(alpha, metrics, verdict, failure_reason)
    print()
    print(reflection)
    print()

    # Save to memory
    print("[ Step 6 ] Saving to database...")
    record = ExperimentRecord(
        alpha_id=alpha["alpha_id"],
        parent_id=alpha.get("parent_id"),
        timestamp=datetime.now(timezone.utc).isoformat(),
        hypothesis=alpha.get("hypothesis", ""),
        formula=alpha.get("formula", ""),
        features=alpha.get("features", []),
        mutation=alpha.get("mutation", ""),
        config=alpha,
        metrics=metrics,
        robustness=robustness,
        verdict=verdict,
        failure_reason=failure_reason,
        reflection=reflection,
    )
    store = ExperimentStore(db_path=args.db)
    store.save_experiment(record)
    print(f"  Saved experiment '{alpha['alpha_id']}' to {args.db}\n")

    print(f"{'='*60}")
    print(f"Done. Run `python scripts/export_graph.py` to visualize the research graph.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
