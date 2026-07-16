# AlphaResearchBot — Version 4.1 Plan
- 再搞明白这个逻辑，什么posterior之类的
- 然后跑个loop试试看，看多少个iteration可以找到promising alpha，再做决定

### decision.py
- removed robustness from objective
- feel like currently we are going to harsh over complexity measure, want to smooth it

### similarity.py
- improved calculation of similarity


- Instead of using a fixed temperature during hypothesis generation, dynamically adjust exploration according to the 
scheduler's decision so that exploration uses higher randomness while local mutation remains more conservative.
- 不光是temperature, weights given to formula novelty and simplicity should also be adjustable given exploration vs exploitation?
- 其实感觉不太对，我这个分数是用来决定下一个action走那边比较好，如果选择exlore就调高novelty的weights，选择exploit就调低novelty的weights感觉不太对。
因为本质上这个分数是post-hoc，不应该因为所选的方向进行灵活调整。

### Next steps
- Rather than treating each alpha as an arm, treat each research direction as an arm (eg. value, profitability, momentum, etc),
and decide which existing research direction deserves more budget
- Currently, AlphaResearchBot primarily implements persistent memory. A future version should instead build a knowledge 
extraction layer that synthesizes observations across many experiments into reusable research knowledge.


### 7.10 Identified issue on duplicate abort when running loop
Cause: After changing the signal evaluation cut-off from a hard cutoff on negative sharpe and negative IC to check on abs(Sharpe),
The current problem came from mixing two separate ideas: 1) Bidirectional evaluation 2) Mutation by sign flip. 
First is useful, the second is redundant.
It created new issues where mutator would suggest flipping the sign of a formula as a mutation, but that won't pass the 
similarity check against previous formulas because operator isn't taken into consideration? (might need to fix this as well).
1. Hence, when running a loop, it always encounters duplicate abort where LLM proposes inverting signs.
- Fix: keep bidirectional scoring, but remove sign-only mutations and REVISE_INVERT from the verdicts, 3 verdicts remain: promising, revise, failed.
2. Check if operator is taken into consideration when doing AST similarity check
3. I already observe so many REVISE_INVERT nodes after running a few experiments, identify the potential cause behind this.
   - Could it be LLM proposing so many hypothesis in the wrong directions? Or is it caused by a structural design issue 
   Potential reasons:
   1. backtest history too short, so a negative sample IC does not necessarily establish that the economic mechanism truly operates backward.
   2. Since my fundamental columns are cross-sectionally standardized before formulas are evaluated, a standardized fundamental 
      - Fix: use raw values to calculate signals, and + standardize the resulting ratio afterward
   3. Some hypotheses are naturally ambiguous in directions, e.g.: growth can represent expansion or overinvestment, volatility can represent either risk premium or low-volatility anomaly.
   4. Current scoring gives every formula two chances: even if the raw direction is only slightly worse because of noise, 
   the inverted version wins. A five-point penalty may be too small, especially around the revise threshold.
can be negative simply because it is below the cross-sectional mean, even when the company has positive cash flow, and hence may not match the intuition.
4. Should I change the score calculation to use the abs(Sharpe) instead? 
- No, if two similar alphas both exhibit high magnitude in Sharpe but in different directions, then that's also valuable information
that would be ignored if I use abs.
5. Should I evaluate a signal twice in both directions?
- No, flipping the sign does NOT provide new information, just record direction contradicted is fine.

Solution: 
- Created two set of scores: total_score that uses raw values, predictive_magnitude that uses abs() values. And determines
which one is used for which purpose (verdict evaluation, diagnostics, parent eligibility).

### 7.15 
Changes made:
1. use raw values to calculate signals, and + standardize the resulting ratio afterward
2. similarity.py
   -  Replaced formula sets with AST-aware structural tokens
   -  Removed the config bonus
   -  Extracted features from the formula rather than relying only on config metadata.
3. Still has the issue of experiment abort due to exact same formula, fix

### 7.16 
- Issue: explore aborts on exact-duplicate hypotheses (planner regenerates tried formulas)
- Root cause 1 — the planner prompt never shows previously tried formulas. 
- Root cause 2 — explore duplicates get no session feedback. 
- Root cause 3 — no duplicate pre-check at generation time. 
- Secondary findings from the same batch:
  1. Iteration 2 used `ASSETS_LTM`, which doesn't exist (only income + cash flow are loaded). Unknown column-like
  tokens are only a validation *warning*, so it passed validation and burned a `backtest_error` instead of failing
  fast at generation where the retry loop could have fixed it.
  2. The mutator regenerated one identical formula after a duplicate rejection (`_mut_...004725` ≡ `_mut_...004733`);
  the `avoid_formulas` feedback caught it on the third try — working, but one pull wasted per first offense.
  3. `temperature=0.3` plus a barely-changing prompt between iterations makes verbatim repetition likely — but raising
  temperature alone would make repeats less deterministic, not informed. The structural fix is visibility.

### Fixes
1. Added tried formulas in the planner prompt — `analyze_memory` now returns `tried_formulas` (the 30 most recent, ALL
verdicts, tagged `[failed]`/`[revise]`/`[promising]`); `_build_plan_prompt` renders them under "Already-tested
formulas" with an explicit instruction not to re-propose any of them or a trivial variant (sign flip, re-weighting,
thin wrapper).
2. Explore-path session feedback — `run_loop.py` records duplicate-aborted formulas for both actions (keyed by
`parent_id` for mutate, `EXPLORE_ARM` for explore) and passes them as `avoid_formulas` into `plan_next_research`,
which injects a "REJECTED as duplicates this session" block into the prompt (same mechanism the mutator had).
3. Generation-time duplicate pre-check, both paths:
   - Planner: `plan_next_research` runs every suggestion (LLM and rule-based fallback) through `check_similarity`
   and drops duplicates before they reach the pipeline; `_llm_plan` requests 2 spare suggestions so filtering
   still leaves n. A known duplicate no longer burns a bandit pull or unfairly punishes the explore arm.
   - Mutator: `_llm_mutation`'s retry loop checks each validated candidate and re-asks the LLM with the duplicate
   as feedback (which alpha, what similarity, "a sign flip / re-weighting / thin wrapper is not a new signal")
   within the existing 3-attempt budget; the rule-based fallback is checked via `_ensure_not_duplicate`, which
   raises a clear generation-failure error instead of letting the duplicate abort at Step 2.
4. Unknown columns are now validation ERRORS (was: warnings) — caught at generation time where the LLM retry loop
can fix the formula, instead of surfacing as a `backtest_error` NameError after committing a pull.