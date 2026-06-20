Version 1 (2026-06-18)


I am building a project called AlphaResearchBot: an iterative alpha discovery research system using agentic AI.

Version 1 Goal:
Build a minimal but extensible research loop that can:

1. Propose or load an alpha experiment.
2. Validate the alpha formula syntax.
3. Run a placeholder/local backtest interface.
4. Compute/store metrics.
5. Ask an LLM or mock LLM to generate a short reflection.
6. Save the experiment into a memory database.
7. Maintain a research graph where every alpha is a node with parent, mutation, metrics, failure reason, and reflection.
8. Export the research graph as an HTML visualization.

Please create the initial repo architecture and working prototype.

Core workflow:
0. Research planner:

* Reads previous experiments from memory.
* Suggests the next research direction.
* For V1, this can be rule-based or mock LLM.

1. Hypothesis:
   Example:
   "Stocks with improving profitability and positive momentum outperform within sectors."

2. Alpha definition:
   Represent alpha as JSON:
   {
   "alpha_id": "alpha_001",
   "parent_id": null,
   "hypothesis": "Stocks with improving profitability and positive momentum outperform within sectors.",
   "formula": "rank(EBITDA_MARGIN) + 0.5 * rank(MOM12_1)",
   "features": ["EBITDA_MARGIN", "MOM12_1"],
   "mutation": "Initial quality plus momentum hypothesis",
   "universe": "sp500",
   "start_date": "2018-01-01",
   "end_date": "2024-12-31",
   "neutralization": "sector",
   "rebalance": "monthly",
   "transaction_cost_bps": 5,
   "holding_period_days": 20
   }

3. Formula validator:
   Check:

* Allowed features only.
* Allowed operators only.
* No future-looking fields.
* No excessive nesting.
* No unavailable data columns. You can refer to data_tables.md within data folder to see available columns.
* Return valid/invalid with reasons.

4. Backtest engine:
   For V1, implement a placeholder deterministic backtest that returns mock but realistic metrics.
   Later I will replace it with real Snowflake + local backtesting.

Return metrics:
{
"IC_mean": float,
"ICIR": float,
"Sharpe": float,
"turnover": float,
"max_drawdown": float,
"deflated_sharpe": float,
"noise_risk": "low|medium|high"
}

5. Robustness:
   For V1, mock:

* sector_stability
* subperiod_stability
* market_regime_sharpe
* placebo_score

6. Decision logic:

* If Sharpe < 0.5 or ICIR < 0.3: failed.
* If turnover > 300: failed due to excessive turnover.
* If deflated_sharpe much lower than Sharpe: high noise risk.
* Else promising.

7. Reflection:
   For V1, generate a simple reflection string:

* observation
* possible reason
* failure reason if failed
* next suggested mutation

Important: clearly label the reason as "LLM-generated hypothesis, not validated evidence."

8. Memory:
   Store every experiment in SQLite with:

* alpha_id
* parent_id
* timestamp
* hypothesis
* formula
* features
* mutation
* config
* metrics
* robustness
* verdict
* failure_reason
* reflection

9. Research graph:
   Use networkx.
   Each alpha is a node.
   Edges represent parent → child.
   Node attributes:

* Sharpe
* ICIR
* verdict
* failure_reason
* mutation
* reflection

10. Visualization:
    Export graph to an HTML file using pyvis.
    Node color:

* green = promising
* red = failed
* yellow = inconclusive/high noise risk

Node tooltip should show:

* alpha_id
* hypothesis
* formula
* Sharpe
* ICIR
* deflated Sharpe
* failure reason
* reflection

11. Scripts:
    Implement:

python scripts/run_experiment.py --config experiments/sample_alpha_001.json

This should:

* Load alpha JSON.
* Validate formula.
* Run mock backtest.
* Compute metrics.
* Generate reflection.
* Save to memory.
* Update research graph.

python scripts/export_graph.py

This should:

* Load experiments from SQLite.
* Build graph.
* Export reports/research_graph.html.

Please prioritize clean modular architecture, type hints, docstrings, and readable code. The first version should run end-to-end locally without Snowflake or real LLM calls.
