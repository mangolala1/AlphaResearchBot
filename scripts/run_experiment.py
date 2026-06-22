"""Run a single alpha experiment end-to-end.

Usage:
    python scripts/run_experiment.py --config experiments/sample_alpha_001.json
    python scripts/run_experiment.py --config experiments/sample_alpha_001.json --no-cache
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from core.backtest import run_backtest
from core.data_loader import DataLoader
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
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore parquet cache and force fresh data fetch"
    )
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
    print("[ Step 2 ] Running backtest...")
    loader = DataLoader(no_cache=getattr(args, "no_cache", False))
    try:
        backtest_result = run_backtest(alpha, data_loader=loader)
    except Exception as exc:
        print(f"  ERROR during backtest: {exc}")
        sys.exit(1)

    metrics = backtest_result["metrics"]
    print(f"  IC_mean        : {metrics['IC_mean']:.4f}  (over {len(backtest_result['ic_series'])} periods)")
    print(f"  ICIR           : {metrics['ICIR']:.4f}")
    print(f"  Sharpe         : {metrics['Sharpe']:.4f}")
    print(f"  Deflated Sharpe: {metrics['deflated_sharpe']:.4f}")
    print(f"  Turnover       : {metrics['turnover']:.1f} bps/yr (estimated)")
    print(f"  Max Drawdown   : {metrics['max_drawdown']:.4f}")
    print(f"  Noise Risk     : {metrics['noise_risk']}\n")

    # Robustness
    print("[ Step 3 ] Running robustness checks...")
    robustness = run_robustness(alpha, backtest_result)
    print(f"  Sector Stability      : {robustness['sector_stability']:.4f}")
    print(f"  Subperiod Stability   : {robustness['subperiod_stability']:.4f}")
    print(f"  Market Regime Sharpe  : {robustness['market_regime_sharpe']:.4f}")
    print(f"  Placebo Score         : {robustness['placebo_score']:.4f}\n")

    # Decision
    print("[ Step 4 ] Applying decision logic...")
    verdict, failure_reason = decide(metrics, robustness)
    print(f"  Verdict: {verdict.upper()}")
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
