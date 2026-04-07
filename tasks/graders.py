"""
Graders for the SQL Query Optimizer environment.
Each grader takes (task_def, action) and returns a Reward.
Scores are deterministic and reproducible — no LLM in the grader loop.
"""
import re
from typing import Dict, Any
from env.models import Action, Reward


def _normalize(sql: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r'\s+', ' ', sql.strip().lower())


def _contains_any(sql: str, patterns: list) -> list:
    n = _normalize(sql)
    return [p for p in patterns if p.lower() in n]


def _contains_all(sql: str, clauses: list) -> list:
    n = _normalize(sql)
    return [c for c in clauses if c.lower() in n]


def _missing(sql: str, clauses: list) -> list:
    found = _contains_all(sql, clauses)
    return [c for c in clauses if c not in found]


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style_score(query: str) -> float:
    """
    Style rubric (0.0–1.0):
    - Avoids SELECT *              +0.3
    - Has meaningful aliases       +0.2
    - Uppercase keywords           +0.2
    - No redundant nested SELECTs  +0.2
    - Has comments or clean layout +0.1
    """
    score = 0.0
    n = _normalize(query)
    raw = query.strip()

    if "select *" not in n:
        score += 0.3
    # Check aliases: AS keyword present
    if " as " in n:
        score += 0.2
    # Uppercase keywords check: COUNT raw uppercase occurrences
    keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", "HAVING"]
    found_upper = sum(1 for k in keywords if k in raw)
    if found_upper >= 3:
        score += 0.2
    # Nested select count
    nested_count = n.count("select")
    if nested_count <= 2:
        score += 0.2
    # Comments or newlines (clean formatting)
    if "\n" in raw or "--" in raw:
        score += 0.1

    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# EASY grader
# ─────────────────────────────────────────────────────────────────────────────

def grade_easy(task_def: Dict[str, Any], action: Action) -> Reward:
    query = action.rewritten_query
    anti = task_def["anti_patterns"]
    required = task_def["required_clauses"]

    # Correctness: avoid anti-patterns + have required clauses
    bad_found = _contains_any(query, anti)
    missing_req = _missing(query, required)

    # Partial credit
    correctness = 1.0
    correctness -= 0.3 * len(bad_found)      # -0.3 per anti-pattern
    correctness -= 0.25 * len(missing_req)   # -0.25 per missing clause
    correctness = max(0.0, min(1.0, correctness))

    # Efficiency: no nested subquery on orders, uses JOIN or EXISTS
    n = _normalize(query)
    efficiency = 1.0
    if "select * from orders" in n:
        efficiency -= 0.5
    if n.count("select") > 2:
        efficiency -= 0.3
    efficiency = max(0.0, min(1.0, efficiency))

    style = _style_score(query)

    total = round(0.5 * correctness + 0.3 * efficiency + 0.2 * style, 4)

    return Reward(
        total=total,
        correctness=correctness,
        efficiency=efficiency,
        style=style,
        breakdown={
            "anti_patterns_found": bad_found,
            "missing_required_clauses": missing_req,
            "weights": {"correctness": 0.5, "efficiency": 0.3, "style": 0.2},
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# MEDIUM grader
# ─────────────────────────────────────────────────────────────────────────────

def grade_medium(task_def: Dict[str, Any], action: Action) -> Reward:
    query = action.rewritten_query
    required = task_def["required_clauses"]  # GROUP BY, HAVING, ORDER BY

    n = _normalize(query)

    # Correctness: required clauses present
    missing_req = _missing(query, required)
    # Must filter status = 'completed'
    has_status_filter = "completed" in n
    # Must compute revenue as quantity * unit_price
    has_revenue_calc = ("quantity" in n and "unit_price" in n) or ("sum" in n and "amount" in n)

    correctness = 1.0
    correctness -= 0.25 * len(missing_req)
    if not has_status_filter:
        correctness -= 0.3
    if not has_revenue_calc:
        correctness -= 0.2
    correctness = max(0.0, min(1.0, correctness))

    # Efficiency: no correlated subquery (heuristic: SELECT inside WHERE)
    correlated_pattern = bool(re.search(r'where\s.*\(select', n))
    # Count top-level selects
    select_count = n.count("select")
    has_group_by = "group by" in n

    efficiency = 1.0
    if correlated_pattern:
        efficiency -= 0.5
    if select_count > 3:
        efficiency -= 0.2
    if not has_group_by:
        efficiency -= 0.3
    efficiency = max(0.0, min(1.0, efficiency))

    style = _style_score(query)

    total = round(0.5 * correctness + 0.3 * efficiency + 0.2 * style, 4)

    return Reward(
        total=total,
        correctness=correctness,
        efficiency=efficiency,
        style=style,
        breakdown={
            "missing_required_clauses": missing_req,
            "has_status_filter": has_status_filter,
            "has_revenue_calculation": has_revenue_calc,
            "correlated_subquery_detected": correlated_pattern,
            "weights": {"correctness": 0.5, "efficiency": 0.3, "style": 0.2},
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# HARD grader
# ─────────────────────────────────────────────────────────────────────────────

def grade_hard(task_def: Dict[str, Any], action: Action) -> Reward:
    query = action.rewritten_query
    required = task_def["required_clauses"]  # WITH, RANK(), OVER, LEFT JOIN, COALESCE

    n = _normalize(query)

    missing_req = _missing(query, required)
    # Must handle zero-sales reps: LEFT JOIN or COALESCE
    handles_zero = "left join" in n or ("coalesce" in n and "0" in n)
    # Must have window function for rank
    has_window = "over" in n and ("rank()" in n or "dense_rank()" in n or "row_number()" in n)
    # Must filter won deals
    filters_won = "won" in n
    # Must have percentage computation
    has_pct = ("/" in n or "percent" in n) and ("max" in n or "sum" in n)

    correctness = 1.0
    correctness -= 0.15 * len(missing_req)
    if not handles_zero:
        correctness -= 0.2
    if not has_window:
        correctness -= 0.25
    if not filters_won:
        correctness -= 0.2
    if not has_pct:
        correctness -= 0.1
    correctness = max(0.0, min(1.0, correctness))

    # Efficiency: uses CTE + window instead of triple correlated subquery
    has_cte = n.startswith("with") or "with " in n[:50]
    triple_correlated = n.count("(select") >= 3

    efficiency = 1.0
    if triple_correlated:
        efficiency -= 0.6
    if not has_cte and not has_window:
        efficiency -= 0.3
    efficiency = max(0.0, min(1.0, efficiency))

    style = _style_score(query)

    total = round(0.5 * correctness + 0.3 * efficiency + 0.2 * style, 4)

    return Reward(
        total=total,
        correctness=correctness,
        efficiency=efficiency,
        style=style,
        breakdown={
            "missing_required_clauses": missing_req,
            "handles_zero_sales_reps": handles_zero,
            "uses_window_function": has_window,
            "filters_won_deals": filters_won,
            "has_percentage_computation": has_pct,
            "uses_cte": has_cte,
            "triple_correlated_detected": triple_correlated,
            "weights": {"correctness": 0.5, "efficiency": 0.3, "style": 0.2},
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

GRADERS = {
    "easy_select_optimization":  grade_easy,
    "medium_join_optimization":  grade_medium,
    "hard_complex_optimization": grade_hard,
}


def grade(task_id: str, task_def: Dict[str, Any], action: Action) -> Reward:
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return GRADERS[task_id](task_def, action)
