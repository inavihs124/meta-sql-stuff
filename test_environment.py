"""
tests/test_environment.py
Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from env.environment import SQLOptimizerEnv, TASK_ORDER
from env.models import Action, Observation, Reward, StepResult, StateSnapshot


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return SQLOptimizerEnv()


# ─────────────────────────────────────────────────────────────────────────────
# Basic API tests
# ─────────────────────────────────────────────────────────────────────────────

def test_reset_returns_observation(env):
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert obs.schema_ddl
    assert obs.slow_query
    assert obs.task_description
    assert obs.task_id in TASK_ORDER


def test_reset_with_task_id(env):
    for task_id in TASK_ORDER:
        obs = env.reset(task_id=task_id)
        assert obs.task_id == task_id


def test_state_returns_snapshot(env):
    env.reset()
    s = env.state()
    assert isinstance(s, StateSnapshot)
    assert s.step_count == 0
    assert not s.episode_done


def test_step_returns_step_result(env):
    env.reset()
    action = Action(rewritten_query="SELECT id FROM customers")
    result = env.step(action)
    assert isinstance(result, StepResult)
    assert isinstance(result.reward, Reward)
    assert result.done is True


def test_reward_in_range(env):
    for task_id in TASK_ORDER:
        env.reset(task_id=task_id)
        action = Action(rewritten_query="SELECT 1")
        result = env.step(action)
        r = result.reward
        assert 0.0 <= r.total <= 1.0
        assert 0.0 <= r.correctness <= 1.0
        assert 0.0 <= r.efficiency <= 1.0
        assert 0.0 <= r.style <= 1.0


def test_step_after_done_raises(env):
    env.reset()
    env.step(Action(rewritten_query="SELECT 1"))
    with pytest.raises(RuntimeError, match="Episode is done"):
        env.step(Action(rewritten_query="SELECT 2"))


def test_reset_clears_done(env):
    env.reset()
    env.step(Action(rewritten_query="SELECT 1"))
    env.reset()
    s = env.state()
    assert not s.episode_done
    assert s.step_count == 0


def test_easy_task_has_hints(env):
    obs = env.reset(task_id="easy_select_optimization")
    assert obs.hints is not None
    assert len(obs.hints) > 0


def test_medium_hard_no_hints(env):
    for tid in ["medium_join_optimization", "hard_complex_optimization"]:
        obs = env.reset(task_id=tid)
        assert obs.hints is None


# ─────────────────────────────────────────────────────────────────────────────
# Grader quality tests — good vs bad queries
# ─────────────────────────────────────────────────────────────────────────────

GOOD_EASY = """
SELECT DISTINCT
    c.first_name,
    c.last_name,
    c.email
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.placed_at >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY c.last_name ASC;
"""

BAD_EASY = "SELECT * FROM customers WHERE id IN (SELECT customer_id FROM (SELECT * FROM orders) AS all_orders)"


def test_easy_good_query_scores_higher(env):
    env.reset(task_id="easy_select_optimization")
    good = env.step(Action(rewritten_query=GOOD_EASY)).reward.total

    env.reset(task_id="easy_select_optimization")
    bad = env.step(Action(rewritten_query=BAD_EASY)).reward.total

    assert good > bad, f"Expected good ({good}) > bad ({bad})"


GOOD_MEDIUM = """
SELECT
    c.name AS category_name,
    SUM(oi.quantity * p.unit_price) AS total_revenue,
    COUNT(DISTINCT oi.product_id)   AS distinct_products
FROM categories c
JOIN products p      ON p.category_id = c.id
JOIN order_items oi  ON oi.product_id = p.id
JOIN orders o        ON o.id = oi.order_id
WHERE o.status = 'completed'
GROUP BY c.id, c.name
HAVING SUM(oi.quantity * p.unit_price) > 500
ORDER BY total_revenue DESC;
"""

def test_medium_good_query_scores_higher(env):
    env.reset(task_id="medium_join_optimization")
    good = env.step(Action(rewritten_query=GOOD_MEDIUM)).reward.total

    env.reset(task_id="medium_join_optimization")
    bad = env.step(Action(rewritten_query="SELECT name FROM categories")).reward.total

    assert good > bad


GOOD_HARD = """
WITH yearly_sales AS (
    SELECT
        r.id,
        r.name,
        COALESCE(SUM(d.amount), 0) AS total_sales
    FROM reps r
    LEFT JOIN deals d ON d.rep_id = r.id
        AND d.status = 'won'
        AND EXTRACT(YEAR FROM d.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE)
    GROUP BY r.id, r.name
),
ranked AS (
    SELECT
        name,
        total_sales,
        RANK() OVER (ORDER BY total_sales DESC) AS rank,
        ROUND(
            total_sales * 100.0 / NULLIF(MAX(total_sales) OVER (), 0), 2
        ) AS pct_of_top
    FROM yearly_sales
)
SELECT name, total_sales, rank, pct_of_top
FROM ranked
ORDER BY rank ASC;
"""

def test_hard_good_query_scores_higher(env):
    env.reset(task_id="hard_complex_optimization")
    good = env.step(Action(rewritten_query=GOOD_HARD)).reward.total

    env.reset(task_id="hard_complex_optimization")
    bad = env.step(Action(rewritten_query="SELECT name FROM reps")).reward.total

    assert good > bad


def test_hard_good_query_above_threshold(env):
    env.reset(task_id="hard_complex_optimization")
    result = env.step(Action(rewritten_query=GOOD_HARD))
    assert result.reward.total >= 0.6, f"Hard good query scored {result.reward.total}, expected >= 0.6"


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_graders_are_deterministic(env):
    """Same query should always produce the same score."""
    for task_id in TASK_ORDER:
        scores = []
        for _ in range(3):
            env.reset(task_id=task_id)
            r = env.step(Action(rewritten_query=GOOD_EASY)).reward.total
            scores.append(r)
        assert len(set(scores)) == 1, f"Non-deterministic scores for {task_id}: {scores}"
