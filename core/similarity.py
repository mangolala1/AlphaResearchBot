"""Structural similarity — AST-based comparison of alpha formulas.

Deliberately SIGN-BLIND: a top-level negation (`-f`, `-1 * f`, `f * -1`) is
stripped during canonicalization, so pure sign flips remain exact duplicates.
Direction is handled by the scoring system (direction_status); a mirrored
backtest carries no new information.

Duplicate detection and novelty are separate measures:

  - Duplicate abort (strict, structure-only):
        exact canonical match  OR  structural_similarity >= threshold
    where structural similarity is a weighted Jaccard over AST-node multisets
    (labeled nodes plus parent→child edges), so `rank(A / B)` differs from
    `rank(B / A)` and nested repetition like ts_mean(ts_mean(X, 20), 20) is
    visible — things the old identifier-token sets could not see.

  - Novelty (graded, feeds 1 - similarity into the composite score):
        0.4 * feature Jaccard + 0.4 * structural + 0.2 * raw-column overlap
    (weights renormalized when declared features are missing on either side).

Config (universe/rebalance/neutralization) contributes NOTHING: matching
config says two alphas were tested under the same framework, not that the
signals are similar.
"""

from __future__ import annotations

import ast
import re
from collections import Counter

from core.formula_validator import AVAILABLE_RAW_COLUMNS
from core.types import AlphaConfig, SimilarityResult

# Combined research-similarity weights (novelty input)
_W_FEATURES  = 0.4
_W_STRUCTURE = 0.4
_W_COLUMNS   = 0.2

# BinOps where operand order carries no information (A * B == B * A)
_COMMUTATIVE_OPS = (ast.Add, ast.Mult)


# ---------------------------------------------------------------------------
# Canonicalization — strip only the OUTER sign, keep everything else
# ---------------------------------------------------------------------------

def _is_neg_one(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
        return isinstance(node, ast.Constant) and node.value == 1
    return isinstance(node, ast.Constant) and node.value == -1


def _strip_outer_sign(node: ast.AST) -> ast.AST:
    """Peel top-level negations: -f, -1 * f, f * -1 (repeatedly)."""
    while True:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            node = node.operand
        elif (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mult)
            and _is_neg_one(node.left)
        ):
            node = node.right
        elif (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mult)
            and _is_neg_one(node.right)
        ):
            node = node.left
        else:
            return node


def _parse_canonical(formula: str) -> ast.AST | None:
    """Parse to an expression AST with the outer sign stripped; None on failure."""
    try:
        tree = ast.parse(formula, mode="eval").body
    except SyntaxError:
        return None
    return _strip_outer_sign(tree)


def canonical_formula(formula: str) -> str | None:
    """Normalized structural representation — f and -f canonicalize identically."""
    tree = _parse_canonical(formula)
    if tree is None:
        return None
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


# ---------------------------------------------------------------------------
# Structural similarity — AST-node multisets with parent→child edges
# ---------------------------------------------------------------------------

def _node_label(n: ast.AST) -> str:
    if isinstance(n, ast.Name):
        return f"name:{n.id}"
    if isinstance(n, ast.Constant):
        return f"const:{n.value!r}"
    if isinstance(n, ast.Call):
        func = n.func
        if isinstance(func, ast.Name):
            return f"call:{func.id}"
        if isinstance(func, ast.Attribute):
            return f"call:.{func.attr}"
        return "call:?"
    if isinstance(n, ast.Attribute):
        return f"attr:{n.attr}"
    if isinstance(n, ast.BinOp):
        return f"binop:{type(n.op).__name__}"
    if isinstance(n, ast.UnaryOp):
        return f"unary:{type(n.op).__name__}"
    return type(n).__name__.lower()


def _ast_multiset(node: ast.AST) -> Counter:
    """Multiset of labeled nodes plus (parent, field, child) edges.

    Edges keep operand order for non-commutative ops (A / B differs from
    B / A) but drop it for + and * (A * B equals B * A). Being a multiset,
    repeated substructures count multiply — nesting the same call twice is
    visible where a set of tokens was not.
    """
    counts: Counter = Counter()

    def visit(n: ast.AST) -> None:
        parent_label = _node_label(n)
        counts[parent_label] += 1
        commutative = isinstance(n, ast.BinOp) and isinstance(n.op, _COMMUTATIVE_OPS)
        for field, value in ast.iter_fields(n):
            children = value if isinstance(value, list) else [value]
            for child in children:
                if not isinstance(child, ast.AST):
                    continue
                edge_field = (
                    "operand" if commutative and field in ("left", "right") else field
                )
                counts[(parent_label, edge_field, _node_label(child))] += 1
                visit(child)

    visit(node)
    return counts


def _multiset_similarity(a: Counter, b: Counter) -> float:
    """Weighted Jaccard over multisets: sum(min) / sum(max)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    keys = a.keys() | b.keys()
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union


def structural_similarity(formula_a: str, formula_b: str) -> float:
    """Sign-blind structural overlap in [0, 1]; token-set fallback on parse failure."""
    tree_a = _parse_canonical(formula_a)
    tree_b = _parse_canonical(formula_b)
    if tree_a is None or tree_b is None:
        return _jaccard(_formula_tokens(formula_a), _formula_tokens(formula_b))
    return _multiset_similarity(_ast_multiset(tree_a), _ast_multiset(tree_b))


# ---------------------------------------------------------------------------
# Column / feature helpers
# ---------------------------------------------------------------------------

def _formula_tokens(formula: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))


def formula_columns(formula: str) -> set[str]:
    """Raw data columns actually referenced by the formula — deterministic,
    unlike the LLM-declared `features` metadata."""
    return _formula_tokens(formula) & AVAILABLE_RAW_COLUMNS


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_similarity(
    new_alpha: AlphaConfig,
    store: "ExperimentStore",  # noqa: F821 — avoid circular import
    threshold: float = 0.95,
) -> SimilarityResult:
    """Compare new_alpha against all experiments in store.

    is_unique=False (abort) only on an exact canonical match or structural
    similarity >= threshold. similarity_score is the softer combined research
    similarity used for the novelty sub-score.
    """
    records = store.load_all()
    if not records:
        return SimilarityResult(
            is_unique=True,
            is_exact_duplicate=False,
            most_similar_id=None,
            similarity_score=0.0,
            structural_similarity=0.0,
            feature_similarity=0.0,
            reason="No prior experiments to compare against.",
        )

    new_formula = new_alpha.get("formula") or ""
    new_canonical = canonical_formula(new_formula)
    new_tree = _parse_canonical(new_formula)
    new_multiset = _ast_multiset(new_tree) if new_tree is not None else None
    new_tokens = _formula_tokens(new_formula)
    new_columns = new_tokens & AVAILABLE_RAW_COLUMNS
    new_features = set(new_alpha.get("features") or [])

    exact_duplicate = False
    best_structural, best_structural_id = 0.0, None
    best_combined, best_combined_id = 0.0, None
    best_combined_features = 0.0

    for record in records:
        formula = record.get("formula") or ""

        # --- structural (drives the duplicate abort) -------------------------
        tree = _parse_canonical(formula)
        if new_multiset is not None and tree is not None:
            s_struct = _multiset_similarity(new_multiset, _ast_multiset(tree))
            if new_canonical is not None and canonical_formula(formula) == new_canonical:
                exact_duplicate = True
                s_struct = 1.0
        else:
            s_struct = _jaccard(new_tokens, _formula_tokens(formula))

        if s_struct > best_structural:
            best_structural = s_struct
            best_structural_id = record["alpha_id"]

        # --- combined research similarity (drives novelty) -------------------
        s_columns = _jaccard(new_columns, formula_columns(formula))
        existing_features = set(record.get("features") or [])
        if new_features and existing_features:
            s_features = _jaccard(new_features, existing_features)
            combined = (
                _W_FEATURES * s_features
                + _W_STRUCTURE * s_struct
                + _W_COLUMNS * s_columns
            )
        else:
            # No declared features on one side — renormalize over the rest
            s_features = 0.0
            w = _W_STRUCTURE + _W_COLUMNS
            combined = (_W_STRUCTURE * s_struct + _W_COLUMNS * s_columns) / w

        if combined > best_combined:
            best_combined = combined
            best_combined_id = record["alpha_id"]
            best_combined_features = s_features

    is_unique = not (exact_duplicate or best_structural >= threshold)
    most_similar_id = best_combined_id if is_unique else best_structural_id

    if exact_duplicate:
        reason = (
            f"Exact structural duplicate of '{best_structural_id}' after sign "
            "canonicalization — a pure sign flip or identical formula. "
            "Direction is already measured; propose a structurally different signal."
        )
    elif not is_unique:
        reason = (
            f"Structural similarity {best_structural:.0%} to '{best_structural_id}' "
            f"is at or above the {threshold:.0%} duplicate threshold. "
            "Mutate further or use --force to override."
        )
    else:
        reason = (
            f"Most similar prior alpha is '{best_combined_id}' at "
            f"{best_combined:.0%} combined similarity (structural max "
            f"{best_structural:.0%}) — below the duplicate threshold."
        ) if best_combined_id else "No prior experiments found."

    return SimilarityResult(
        is_unique=is_unique,
        is_exact_duplicate=exact_duplicate,
        most_similar_id=most_similar_id,
        similarity_score=round(best_combined, 4),
        structural_similarity=round(best_structural, 4),
        feature_similarity=round(best_combined_features, 4),
        reason=reason,
    )
