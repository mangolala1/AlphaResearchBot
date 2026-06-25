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
from core.decision import check_tier1, decide
from core.memory import ExperimentStore
from core.reflection import generate_reflection
from core.robustness import run_robustness
from core.similarity import check_similarity
from core.types import ExperimentRecord
from core.validator import validate_alpha


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an alpha experiment")
    parser.add_argument("--config", required=True, help="Path to alpha JSON config")
    parser.add_argument("--db", default="db/experiments.db", help="SQLite database path")
    parser.add_argument(
        "--no-cache", action="store_true", help="Ignore parquet cache and force fresh data fetch"
    )
    parser.add_argument(
        "--force", action="store_true", help="Skip similarity check and run even if alpha is too similar to a prior one"
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

    # Similarity check
    print("[ Step 1.5 ] Checking similarity against prior alphas...")
    _store = ExperimentStore(db_path=args.db)
    sim = check_similarity(alpha, _store)
    if not sim["is_unique"]:
        print(f"  WARNING: Alpha is {sim['similarity_score']:.0%} similar to '{sim['most_similar_id']}'")
        print(f"  {sim['reason']}")
        if not args.force:
            print("  Use --force to run anyway.")
            sys.exit(1)
        else:
            print("  --force flag set, continuing.\n")
    else:
        print(f"  {sim['reason']}\n")

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
    print(f"  Monotonicity   : {metrics['monotonicity']:.4f}")
    print(f"  Sharpe         : {metrics['Sharpe']:.4f}")
    print(f"  Deflated Sharpe: {metrics['deflated_sharpe']:.4f}")
    print(f"  Turnover       : {metrics['turnover']:.4f}  (constituent replacement rate)")
    print(f"  Max Drawdown   : {metrics['max_drawdown']:.4f}")
    print(f"  Noise Risk     : {metrics['noise_risk']}\n")

    # Tier 1 check — early exit if no predictive power
    print("[ Step 3 ] Tier 1 — predictive power check...")
    tier1_verdict, tier1_reason = check_tier1(metrics)
    if tier1_verdict == "failed":
        print(f"  FAILED: {tier1_reason}\n")
        verdict, failure_reason = "failed", tier1_reason
        robustness = {"sector_stability": {}, "subperiod_stability": 0.0,
                      "market_regime_sharpe": {}, "placebo_score": 0.0}
    else:
        if tier1_verdict == "revise":
            print(f"  WEAK: {tier1_reason}")
        else:
            print("  Passed.")
        print()

        # Robustness (Tier 3 data)
        print("[ Step 4 ] Running robustness checks (Tier 3 diagnostics)...")
        robustness = run_robustness(alpha, backtest_result)
        sector_summary = "  ".join(
            f"{s}: {v:+.3f}" for s, v in sorted(
                robustness["sector_stability"].items(), key=lambda x: -abs(x[1])
            )[:5]
        )
        regime_summary = "  ".join(
            f"{r}: {v:+.3f}" for r, v in robustness["market_regime_sharpe"].items()
        )
        print(f"  Sector IC          : {sector_summary or 'N/A'}")
        print(f"  Subperiod Stability: {robustness['subperiod_stability']:.4f}")
        print(f"  Regime Sharpes     : {regime_summary or 'N/A'}")
        print(f"  Placebo Score      : {robustness['placebo_score']:.4f}\n")

        # Tier 2 decision
        print("[ Step 5 ] Tier 2 — implementation check...")
        verdict, failure_reason = decide(metrics, robustness)
        # If Tier 1 was soft and Tier 2 says promising, keep as revise
        if tier1_verdict == "revise" and verdict == "promising":
            verdict, failure_reason = "revise", tier1_reason
        print(f"  Verdict: {verdict.upper()}")
        if failure_reason:
            print(f"  Reason : {failure_reason}")
        print()

    # Reflection
    print("[ Step 6 ] Generating reflection...")
    reflection = generate_reflection(alpha, metrics, robustness, verdict, failure_reason)
    print()
    print(reflection)
    print()

    # Save to memory
    print("[ Step 7 ] Saving to database...")
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
