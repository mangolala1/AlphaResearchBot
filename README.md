# AlphaResearchBot

An agentic system for autonomous quantitative alpha research on US equities.

## Motivation

1. Quantitative alpha research is fundamentally an **iterative process**: form a hypothesis, build a signal, backtest it, diagnose why it failed, and try again. In practice this loop is slow and manual. A researcher might spend days writing a signal, running a backtest, and interpreting the output — only to discover the alpha had excessive turnover or an inconsistent IC across sectors. Then they start over.

2. The **search space is enormous**, and most signals are dead ends. A researcher might try hundreds of combinations of features, weights, and rebalance frequencies before finding a signal that works. By storing every experiment and its verdict, the system can avoid re-testing near-duplicate alphas, learn which types of signals fail in which ways, and generate next steps that are grounded in evidence rather than intuition alone.

The core insight behind this project is that most of that loop can be automated. The hypothesis-to-verdict pipeline is well-defined enough to codify, and the "what went wrong and what should I try next" question is exactly the kind of reasoning that LLMs are good at. If you can close the loop — so that a failed experiment automatically informs the next one — you get a research process that compounds on itself rather than restarting from scratch each time.

AlphaResearchBot is a proof of concept for that idea. It maintains a persistent memory of every experiment it has run, uses an LLM to reflect on failures and propose new directions, and tracks the full lineage of hypotheses as a tree so you can see how ideas evolved. The goal is not to replace the researcher, but to handle the mechanical parts of the loop so the researcher can focus on the ideas themselves.

## The Research Loop

The system runs a closed cycle:

1. **Hypothesize** — define an investment thesis and express it as a formula over financial features (e.g. `rank(EBITDA_MARGIN) + rank(MOM12_1)`)
2. **Backtest** — run a monthly-rebalanced long-short backtest on S&P 500 data, measuring IC, Sharpe, turnover, and drawdown
3. **Stress-test** — check whether the signal holds across sectors, market regimes, and subperiods, and whether it survives a placebo test
4. **Decide** — apply hard thresholds to classify the experiment as `promising`, `failed`, or `inconclusive`
5. **Reflect** — ask an LLM to diagnose the failure mode and identify what specifically to change
6. **Evolve** — either plan a new branch of research based on the full experiment history, or mutate the failed alpha to address its specific weakness
7. **Remember** — store everything (metrics, reflection, lineage) so future experiments can learn from it

Each experiment links back to its parent, building a research tree over time. This means you can trace exactly why an alpha exists and what it was trying to fix.

## Alpha Evolution Graph

Each run is a node in a growing research tree. Failed experiments are mutated into children that address the specific failure mode, while promising ones branch into new directions.

Research progresses in **rings**: each call to `plan_next.py` generates one batch of alpha ideas that form a concentric ring. The innermost ring is the first batch; each subsequent `plan_next.py` run adds a surrounding ring. Mutations can cross rings freely and are shown as bold orange directed edges. Node color indicates verdict: red = failed, orange = revise, green = promising.

Clicking any node opens a side panel with three tiers of diagnostics:
- **Tier 1 — Predictive Power**: IC mean, ICIR, monotonicity
- **Tier 2 — Implementation**: Sharpe, turnover, max drawdown
- **Tier 3 — Diagnostics**: IC by industry (top 4 sectors), subperiod stability, market regime Sharpe (bull/bear/high_vol/low_vol), and placebo score

Below is a sample graph of a few experiments, showing their hypotheses, features, and Sharpe ratios.
![Example](Images%2FScreenshot%202026-06-28%20at%2022.45.10.png)

## Why This Matters

Most backtesting tools treat each experiment as independent. You run a signal, get a number, move on. There's no memory of what you've already tried, no systematic diagnosis of failures, and no mechanism to ensure the next experiment is meaningfully different from the last. Researchers end up rediscovering the same dead ends.

By giving the system a persistent experiment store and LLM-powered planning, AlphaResearchBot can avoid re-testing near-duplicate alphas, learn which types of signals fail in which ways, and generate next steps that are grounded in evidence rather than intuition alone.

## Roadmap

The project is built in stages, each proving a different capability:

| Version   | Goal |
|-----------|---|
| V1        | Prove the architecture with a fully mock pipeline |
| V2        | Replace mocks with real market data (yfinance, SimFin) and a real backtest engine |
| V3        | Add real LLM reflection, a research planner, and a full agentic loop |
| V3.5      | Add alpha mutation and an interactive research graph |
| V3.9      | Ring layout with batch tracking, graph filtering flags, backtest trading-date fix, LLM formula self-correction, robustness diagnostics in graph |
| V4 (plan) | MCP integration, literature retrieval, autonomous validation agent |

## Quickstart

```bash
# Set up
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run an experiment
python scripts/run_experiment.py --config experiments/sample_alpha_001.json

# Let the system plan what to try next
python scripts/plan_next.py --n 3 --save

# Mutate a failed experiment
python scripts/mutate_alpha.py --parent alpha_001 --run

# Visualize the full research tree
python scripts/export_graph.py && open reports/research_graph.html

# Filter the graph (flags can be combined)
python scripts/export_graph.py --filter-verdict promising
python scripts/export_graph.py --filter-top 10
python scripts/export_graph.py --filter-batch batch_20260628_120000
python scripts/export_graph.py --filter-verdict promising --include-ancestors
```

See [COMMANDS.md](COMMANDS.md) for full usage, config options, and supported alpha features.
