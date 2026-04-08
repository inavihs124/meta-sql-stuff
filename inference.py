"""
inference.py — Baseline Inference Script for SQL Query Optimizer OpenEnv
=========================================================================
MANDATORY (validator requirements):
  - Uses os.environ["API_BASE_URL"] and os.environ["API_KEY"] directly
  - OpenAI client initialized with base_url=os.environ["API_BASE_URL"]
    and api_key=os.environ["API_KEY"]
  - Emits [START] / [STEP] / [END] structured stdout
  - MODEL_NAME read from os.environ["MODEL_NAME"]
"""
import os
import sys
import json
import time

# ── Config (validator injects these; fallbacks are for local testing only) ────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", ""))
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://shivanims-meta-sql-stuff.hf.space")

# Fail-fast guard: if API_BASE_URL or API_KEY are missing, print a clear error
if not API_BASE_URL:
    print("[ERROR] API_BASE_URL environment variable is not set.", flush=True)
    sys.exit(1)
if not API_KEY:
    print("[ERROR] API_KEY environment variable is not set.", flush=True)
    sys.exit(1)

TASK_IDS = [
    "easy_select_optimization",
    "medium_join_optimization",
    "hard_complex_optimization",
]

# ── Hardcoded task observations (fallback when env server unreachable) ────────
FALLBACK_OBSERVATIONS = {
    "easy_select_optimization": {
        "task_description": (
            "Return the full name and email of all customers who placed at least one "
            "order in the last 30 days. Order results by last name ascending."
        ),
        "schema_ddl": """\
CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    first_name  TEXT NOT NULL,
    last_name   TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    total       NUMERIC(10,2) NOT NULL,
    placed_at   TIMESTAMP NOT NULL
);""",
        "slow_query": """\
SELECT * FROM customers
WHERE id IN (
    SELECT customer_id FROM (
        SELECT * FROM orders
    ) AS all_orders
    WHERE all_orders.placed_at >= CURRENT_DATE - INTERVAL '30 days'
);""",
        "hints": [
            "Use a JOIN instead of a nested subquery",
            "Select only first_name, last_name, email — not SELECT *",
            "Use DISTINCT to avoid duplicate customers",
            "Add ORDER BY last_name ASC",
        ],
        "sample_data": {
            "customers": [
                {"id": 1, "first_name": "Alice", "last_name": "Smith",  "email": "alice@example.com"},
                {"id": 2, "first_name": "Bob",   "last_name": "Jones",  "email": "bob@example.com"},
                {"id": 3, "first_name": "Carol",  "last_name": "Adams", "email": "carol@example.com"},
            ],
            "orders": [
                {"id": 1, "customer_id": 1, "total": 99.99,  "placed_at": "2025-03-25 10:00:00"},
                {"id": 2, "customer_id": 2, "total": 149.50, "placed_at": "2024-01-01 08:00:00"},
            ],
        },
    },
    "medium_join_optimization": {
        "task_description": (
            "For each product category, return the category name, total revenue "
            "(sum of quantity * unit_price for completed orders), and the number of distinct "
            "products sold. Only include categories with total revenue above $500. "
            "Order by total revenue descending."
        ),
        "schema_ddl": """\
CREATE TABLE categories (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL
);

CREATE TABLE products (
    id          INTEGER PRIMARY KEY,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    name        TEXT NOT NULL,
    unit_price  NUMERIC(10,2) NOT NULL
);

CREATE TABLE orders (
    id         INTEGER PRIMARY KEY,
    status     TEXT NOT NULL,
    placed_at  TIMESTAMP NOT NULL
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL
);""",
        "slow_query": """\
SELECT
    c.name,
    (SELECT SUM(oi.quantity * p2.unit_price)
     FROM order_items oi
     JOIN products p2 ON oi.product_id = p2.id
     JOIN orders o2   ON oi.order_id   = o2.id
     WHERE p2.category_id = c.id AND o2.status = 'completed') AS total_revenue,
    (SELECT COUNT(DISTINCT oi2.product_id)
     FROM order_items oi2
     JOIN products p3 ON oi2.product_id = p3.id
     JOIN orders o3   ON oi2.order_id   = o3.id
     WHERE p3.category_id = c.id AND o3.status = 'completed') AS distinct_products
FROM categories c
ORDER BY total_revenue DESC;""",
        "hints": None,
        "sample_data": {
            "categories": [
                {"id": 1, "name": "Electronics"},
                {"id": 2, "name": "Books"},
                {"id": 3, "name": "Clothing"},
            ],
            "products": [
                {"id": 1, "category_id": 1, "name": "Laptop",   "unit_price": 999.99},
                {"id": 2, "category_id": 1, "name": "Phone",    "unit_price": 699.99},
                {"id": 3, "category_id": 2, "name": "Novel",    "unit_price": 14.99},
                {"id": 4, "category_id": 3, "name": "T-Shirt",  "unit_price": 29.99},
            ],
            "orders": [
                {"id": 1, "status": "completed", "placed_at": "2025-03-01"},
                {"id": 2, "status": "cancelled", "placed_at": "2025-03-05"},
            ],
            "order_items": [
                {"id": 1, "order_id": 1, "product_id": 1, "quantity": 2},
                {"id": 2, "order_id": 1, "product_id": 3, "quantity": 5},
                {"id": 3, "order_id": 2, "product_id": 4, "quantity": 3},
            ],
        },
    },
    "hard_complex_optimization": {
        "task_description": (
            "For each sales representative, compute: their name, total sales amount for "
            "the current calendar year, their rank among all reps by total sales (1 = highest), "
            "and the percentage their sales represent of the top performer's sales. "
            "Include reps with zero sales. Order by rank ascending."
        ),
        "schema_ddl": """\
CREATE TABLE reps (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    region     TEXT NOT NULL,
    hired_at   DATE NOT NULL
);

CREATE TABLE deals (
    id          INTEGER PRIMARY KEY,
    rep_id      INTEGER NOT NULL REFERENCES reps(id),
    amount      NUMERIC(12,2) NOT NULL,
    closed_at   DATE NOT NULL,
    status      TEXT NOT NULL
);""",
        "slow_query": """\
SELECT
    r.name,
    (SELECT COALESCE(SUM(d.amount), 0)
     FROM deals d
     WHERE d.rep_id = r.id
       AND d.status = 'won'
       AND EXTRACT(YEAR FROM d.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE)) AS total_sales,
    (SELECT COUNT(*) + 1
     FROM reps r2
     WHERE (SELECT COALESCE(SUM(d2.amount), 0)
            FROM deals d2
            WHERE d2.rep_id = r2.id
              AND d2.status = 'won'
              AND EXTRACT(YEAR FROM d2.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE))
           >
           (SELECT COALESCE(SUM(d3.amount), 0)
            FROM deals d3
            WHERE d3.rep_id = r.id
              AND d3.status = 'won'
              AND EXTRACT(YEAR FROM d3.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE))
    ) AS rank
FROM reps r
ORDER BY rank ASC;""",
        "hints": None,
        "sample_data": {
            "reps": [
                {"id": 1, "name": "Diana Prince", "region": "West",  "hired_at": "2020-01-15"},
                {"id": 2, "name": "Bruce Wayne",  "region": "East",  "hired_at": "2019-06-01"},
                {"id": 3, "name": "Clark Kent",   "region": "North", "hired_at": "2021-03-10"},
                {"id": 4, "name": "Lois Lane",    "region": "South", "hired_at": "2022-08-22"},
            ],
            "deals": [
                {"id": 1, "rep_id": 1, "amount": 50000.00, "closed_at": "2025-02-14", "status": "won"},
                {"id": 2, "rep_id": 1, "amount": 30000.00, "closed_at": "2025-03-01", "status": "won"},
                {"id": 3, "rep_id": 2, "amount": 90000.00, "closed_at": "2025-01-20", "status": "won"},
                {"id": 4, "rep_id": 3, "amount": 20000.00, "closed_at": "2024-12-15", "status": "won"},
                {"id": 5, "rep_id": 3, "amount": 15000.00, "closed_at": "2025-02-28", "status": "lost"},
            ],
        },
    },
}

SYSTEM_PROMPT = """You are an expert SQL engineer.
You will be given a database schema, a slow/poorly-written SQL query, and a task description.
Your job is to rewrite the query to be:
1. Semantically CORRECT — it must compute exactly what the task asks
2. EFFICIENT — eliminate redundant subqueries, use proper JOINs, GROUP BY, window functions
3. READABLE — uppercase keywords, meaningful aliases, no SELECT *

Respond with ONLY the rewritten SQL query, no explanation, no markdown fences."""


# ── Logging helpers ───────────────────────────────────────────────────────────
def _emit(obj: dict):
    event = obj.get("event", "")

    if event == "[START]":
        tasks_str = ",".join(obj.get("tasks", []))
        print(
            f"[START] task={tasks_str} model={obj.get('model', '')} "
            f"env_url={obj.get('env_url', '')}",
            flush=True,
        )

    elif event == "[STEP]":
        task_id  = obj.get("task_id", "unknown")
        score    = obj.get("score", 0.0)
        step_num = obj.get("step", 1)
        error    = obj.get("error", "")
        if error:
            print(f"[STEP] step={step_num} task={task_id} reward={score} error={error}", flush=True)
        else:
            print(
                f"[STEP] step={step_num} task={task_id} reward={score} "
                f"correctness={obj.get('correctness', '')} "
                f"efficiency={obj.get('efficiency', '')} "
                f"style={obj.get('style', '')} "
                f"latency_s={obj.get('latency_s', '')}",
                flush=True,
            )

    elif event == "[END]":
        scores     = obj.get("scores", {})
        mean_score = obj.get("mean_score", 0.0)
        n_steps    = len(scores)
        for task_id, score in scores.items():
            print(f"[END] task={task_id} score={score} steps=1", flush=True)
        print(
            f"[END] task=ALL score={mean_score} steps={n_steps} "
            f"model={obj.get('model', '')}",
            flush=True,
        )

    elif event == "[ERROR]":
        print(
            f"[ERROR] stage={obj.get('stage', '')} error={obj.get('error', '')}",
            flush=True,
        )

    # Also emit raw JSON for debugging
    print(json.dumps(obj), flush=True)


# ── Environment helpers ───────────────────────────────────────────────────────
def env_reset(task_id: str) -> dict:
    import requests
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(rewritten_query: str) -> dict:
    import requests
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"rewritten_query": rewritten_query},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_user_prompt(obs: dict) -> str:
    hints_section = ""
    if obs.get("hints"):
        hints_section = "\n\nHINTS:\n" + "\n".join(f"- {h}" for h in obs["hints"])

    sample_section = ""
    if obs.get("sample_data"):
        sample_section = "\n\nSAMPLE DATA:\n"
        for table, rows in obs["sample_data"].items():
            sample_section += f"\n{table}:\n"
            for row in rows[:3]:
                sample_section += f"  {row}\n"

    return (
        f"TASK: {obs['task_description']}\n\n"
        f"SCHEMA:\n{obs['schema_ddl']}\n\n"
        f"SLOW QUERY TO OPTIMIZE:\n{obs['slow_query']}"
        f"{hints_section}{sample_section}\n\nRewrite this query:"
    )


# ── Core task runner ──────────────────────────────────────────────────────────
def run_task(client, task_id: str) -> dict:
    # Try live env server; fall back to hardcoded observations
    env_available = True
    try:
        obs = env_reset(task_id)
    except Exception as exc:
        print(f"[DEBUG] env_reset failed for {task_id}: {exc}", flush=True)
        obs = FALLBACK_OBSERVATIONS[task_id]
        env_available = False

    user_prompt = build_user_prompt(obs)
    start = time.time()

    # ── THE MANDATORY LLM CALL via validator's proxy ──────────────────────────
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    latency = round(time.time() - start, 2)
    rewritten_query = completion.choices[0].message.content.strip()

    if not env_available:
        # LLM was called (proxy hit registered); grading returns 0 since no env
        return {
            "task_id":         task_id,
            "rewritten_query": rewritten_query,
            "reward":          {"total": 0.0, "correctness": 0.0, "efficiency": 0.0, "style": 0.0},
            "latency_s":       latency,
        }

    result = env_step(rewritten_query)
    reward = result.get("reward", {"total": 0.0, "correctness": 0.0, "efficiency": 0.0, "style": 0.0})
    return {
        "task_id":         task_id,
        "rewritten_query": rewritten_query,
        "reward":          reward,
        "latency_s":       latency,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # [START] — always the very first line emitted
    _emit({
        "event":   "[START]",
        "model":   MODEL_NAME,
        "tasks":   TASK_IDS,
        "env_url": ENV_BASE_URL,
    })

    all_scores: list[float] = []

    # Initialise OpenAI client exactly as the validator requires:
    #   base_url = os.environ["API_BASE_URL"]
    #   api_key  = os.environ["API_KEY"]
    from openai import OpenAI
    client = OpenAI(
        base_url=os.environ.get("API_BASE_URL", API_BASE_URL),
        api_key=os.environ.get("API_KEY", API_KEY),
    )

    # Per-task loop
    for step_num, task_id in enumerate(TASK_IDS, start=1):
        try:
            result = run_task(client, task_id)
            reward = result["reward"]
            score  = reward.get("total", 0.0)
            all_scores.append(score)
            rq = result["rewritten_query"]
            _emit({
                "event":           "[STEP]",
                "step":            step_num,
                "task_id":         task_id,
                "score":           score,
                "correctness":     reward.get("correctness", 0.0),
                "efficiency":      reward.get("efficiency", 0.0),
                "style":           reward.get("style", 0.0),
                "latency_s":       result["latency_s"],
                "rewritten_query": rq[:200] + "..." if len(rq) > 200 else rq,
            })
        except Exception as e:
            print(f"[DEBUG] run_task raised for {task_id}: {e}", flush=True)
            all_scores.append(0.0)
            _emit({
                "event":   "[STEP]",
                "step":    step_num,
                "task_id": task_id,
                "score":   0.0,
                "error":   str(e),
            })

    mean_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    # [END] — always emitted
    _emit({
        "event":      "[END]",
        "scores":     {tid: s for tid, s in zip(TASK_IDS, all_scores)},
        "mean_score": mean_score,
        "model":      MODEL_NAME,
    })


if __name__ == "__main__":
    main()
