"""
inference.py — Baseline Inference Script for SQL Query Optimizer OpenEnv
Reads: API_KEY, API_BASE_URL, MODEL_NAME, ENV_BASE_URL  (injected by validator)
Emits structured stdout: [START], [STEP], [END] format (required by OpenEnv bootcamp)

Usage:
    API_KEY=sk-... MODEL_NAME=gpt-4o python inference.py
"""
import os
import sys
import json
import time

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://shivanims-meta-sql-stuff.hf.space")

TASK_IDS = [
    "easy_select_optimization",
    "medium_join_optimization",
    "hard_complex_optimization",
]

FALLBACK_OBSERVATIONS = {
    "easy_select_optimization": {
        "task_description": "Return the full name and email of all customers who placed at least one order in the last 30 days. Order results by last name ascending.",
        "schema_ddl": "CREATE TABLE customers (\n    id          INTEGER PRIMARY KEY,\n    first_name  TEXT NOT NULL,\n    last_name   TEXT NOT NULL,\n    email       TEXT NOT NULL UNIQUE,\n    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);\n\nCREATE TABLE orders (\n    id          INTEGER PRIMARY KEY,\n    customer_id INTEGER NOT NULL REFERENCES customers(id),\n    total       NUMERIC(10,2) NOT NULL,\n    placed_at   TIMESTAMP NOT NULL\n);",
        "slow_query": "SELECT * FROM customers\nWHERE id IN (\n    SELECT customer_id FROM (\n        SELECT * FROM orders\n    ) AS all_orders\n    WHERE all_orders.placed_at >= CURRENT_DATE - INTERVAL '30 days'\n);",
        "hints": [
            "Use a JOIN instead of a nested subquery",
            "Select only first_name, last_name, email — not SELECT *",
            "Use DISTINCT to avoid duplicate customers",
            "Add ORDER BY last_name ASC",
        ],
        "sample_data": {
            "customers": [
                {"id": 1, "first_name": "Alice", "last_name": "Smith", "email": "alice@example.com"},
                {"id": 2, "first_name": "Bob",   "last_name": "Jones", "email": "bob@example.com"},
                {"id": 3, "first_name": "Carol",  "last_name": "Adams", "email": "carol@example.com"},
            ],
            "orders": [
                {"id": 1, "customer_id": 1, "total": 99.99,  "placed_at": "2025-03-25 10:00:00"},
                {"id": 2, "customer_id": 2, "total": 149.50, "placed_at": "2024-01-01 08:00:00"},
            ]
        }
    },
    "medium_join_optimization": {
        "task_description": "For each product category, return the category name, total revenue (sum of quantity * unit_price for completed orders), and the number of distinct products sold. Only include categories with total revenue above $500. Order by total revenue descending.",
        "schema_ddl": "CREATE TABLE categories (\n    id    INTEGER PRIMARY KEY,\n    name  TEXT NOT NULL\n);\n\nCREATE TABLE products (\n    id          INTEGER PRIMARY KEY,\n    category_id INTEGER NOT NULL REFERENCES categories(id),\n    name        TEXT NOT NULL,\n    unit_price  NUMERIC(10,2) NOT NULL\n);\n\nCREATE TABLE orders (\n    id         INTEGER PRIMARY KEY,\n    status     TEXT NOT NULL,   -- 'pending','completed','cancelled'\n    placed_at  TIMESTAMP NOT NULL\n);\n\nCREATE TABLE order_items (\n    id         INTEGER PRIMARY KEY,\n    order_id   INTEGER NOT NULL REFERENCES orders(id),\n    product_id INTEGER NOT NULL REFERENCES products(id),\n    quantity   INTEGER NOT NULL\n);",
        "slow_query": "SELECT\n    c.name,\n    (SELECT SUM(oi.quantity * p2.unit_price)\n     FROM order_items oi\n     JOIN products p2 ON oi.product_id = p2.id\n     JOIN orders o2   ON oi.order_id   = o2.id\n     WHERE p2.category_id = c.id AND o2.status = 'completed') AS total_revenue,\n    (SELECT COUNT(DISTINCT oi2.product_id)\n     FROM order_items oi2\n     JOIN products p3 ON oi2.product_id = p3.id\n     JOIN orders o3   ON oi2.order_id   = o3.id\n     WHERE p3.category_id = c.id AND o3.status = 'completed') AS distinct_products\nFROM categories c\nORDER BY total_revenue DESC;",
        "hints": None,
        "sample_data": {
            "categories": [{"id": 1, "name": "Electronics"}, {"id": 2, "name": "Books"}, {"id": 3, "name": "Clothing"}],
            "products": [{"id": 1, "category_id": 1, "name": "Laptop", "unit_price": 999.99}, {"id": 2, "category_id": 1, "name": "Phone", "unit_price": 699.99}, {"id": 3, "category_id": 2, "name": "Novel", "unit_price": 14.99}, {"id": 4, "category_id": 3, "name": "T-Shirt", "unit_price": 29.99}],
            "orders": [{"id": 1, "status": "completed", "placed_at": "2025-03-01"}, {"id": 2, "status": "cancelled", "placed_at": "2025-03-05"}],
            "order_items": [{"id": 1, "order_id": 1, "product_id": 1, "quantity": 2}, {"id": 2, "order_id": 1, "product_id": 3, "quantity": 5}, {"id": 3, "order_id": 2, "product_id": 4, "quantity": 3}]
        }
    },
    "hard_complex_optimization": {
        "task_description": "For each sales representative, compute: their name, total sales amount for the current calendar year, their rank among all reps by total sales (1 = highest), and the percentage their sales represent of the top performer's sales. Include reps with zero sales. Order by rank ascending.",
        "schema_ddl": "CREATE TABLE reps (\n    id         INTEGER PRIMARY KEY,\n    name       TEXT NOT NULL,\n    region     TEXT NOT NULL,\n    hired_at   DATE NOT NULL\n);\n\nCREATE TABLE deals (\n    id          INTEGER PRIMARY KEY,\n    rep_id      INTEGER NOT NULL REFERENCES reps(id),\n    amount      NUMERIC(12,2) NOT NULL,\n    closed_at   DATE NOT NULL,\n    status      TEXT NOT NULL   -- 'won','lost','open'\n);",
        "slow_query": "SELECT\n    r.name,\n    (SELECT COALESCE(SUM(d.amount), 0)\n     FROM deals d\n     WHERE d.rep_id = r.id\n       AND d.status = 'won'\n       AND EXTRACT(YEAR FROM d.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE)) AS total_sales,\n    (SELECT COUNT(*) + 1\n     FROM reps r2\n     WHERE (SELECT COALESCE(SUM(d2.amount), 0)\n            FROM deals d2\n            WHERE d2.rep_id = r2.id\n              AND d2.status = 'won'\n              AND EXTRACT(YEAR FROM d2.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE))\n           >\n           (SELECT COALESCE(SUM(d3.amount), 0)\n            FROM deals d3\n            WHERE d3.rep_id = r.id\n              AND d3.status = 'won'\n              AND EXTRACT(YEAR FROM d3.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE))\n    ) AS rank\nFROM reps r\nORDER BY rank ASC;",
        "hints": None,
        "sample_data": {
            "reps": [{"id": 1, "name": "Diana Prince", "region": "West", "hired_at": "2020-01-15"}, {"id": 2, "name": "Bruce Wayne", "region": "East", "hired_at": "2019-06-01"}, {"id": 3, "name": "Clark Kent", "region": "North", "hired_at": "2021-03-10"}, {"id": 4, "name": "Lois Lane", "region": "South", "hired_at": "2022-08-22"}],
            "deals": [{"id": 1, "rep_id": 1, "amount": 50000.00, "closed_at": "2025-02-14", "status": "won"}, {"id": 2, "rep_id": 1, "amount": 30000.00, "closed_at": "2025-03-01", "status": "won"}, {"id": 3, "rep_id": 2, "amount": 90000.00, "closed_at": "2025-01-20", "status": "won"}, {"id": 4, "rep_id": 3, "amount": 20000.00, "closed_at": "2024-12-15", "status": "won"}, {"id": 5, "rep_id": 3, "amount": 15000.00, "closed_at": "2025-02-28", "status": "lost"}]
        }
    }
}

SYSTEM_PROMPT = """You are an expert SQL engineer.
You will be given a database schema, a slow/poorly-written SQL query, and a task description.
Your job is to rewrite the query to be:
1. Semantically CORRECT — it must compute exactly what the task asks
2. EFFICIENT — eliminate redundant subqueries, use proper JOINs, GROUP BY, window functions
3. READABLE — uppercase keywords, meaningful aliases, no SELECT *

Respond with ONLY the rewritten SQL query, no explanation, no markdown fences."""


def _emit(obj: dict):
    """Print structured output to stdout and flush immediately.
    The validator requires lines that START with the literal token:
      [START] task=NAME
      [STEP]  step=N reward=R
      [END]   task=NAME score=S steps=N
    Extra key=value pairs are appended on the same line.
    A JSON detail line is also emitted afterwards for debugging.
    """
    event = obj.get("event", "")

    if event == "[START]":
        tasks_str = ",".join(obj.get("tasks", []))
        print(f"[START] task={tasks_str} model={obj.get('model', '')} env_url={obj.get('env_url', '')}", flush=True)

    elif event == "[STEP]":
        task_id  = obj.get("task_id", "unknown")
        score    = obj.get("score", 0.0)
        error    = obj.get("error", "")
        step_num = obj.get("step", 1)
        if error:
            print(f"[STEP] step={step_num} task={task_id} reward={score} error={error}", flush=True)
        else:
            correctness = obj.get("correctness", "")
            efficiency  = obj.get("efficiency", "")
            style       = obj.get("style", "")
            latency     = obj.get("latency_s", "")
            print(
                f"[STEP] step={step_num} task={task_id} reward={score} "
                f"correctness={correctness} efficiency={efficiency} style={style} latency_s={latency}",
                flush=True,
            )

    elif event == "[END]":
        scores     = obj.get("scores", {})
        mean_score = obj.get("mean_score", 0.0)
        n_steps    = len(scores)
        for task_id, score in scores.items():
            print(f"[END] task={task_id} score={score} steps=1", flush=True)
        print(f"[END] task=ALL score={mean_score} steps={n_steps} model={obj.get('model', '')}", flush=True)

    elif event == "[ERROR]":
        print(f"[ERROR] stage={obj.get('stage','')} error={obj.get('error','')}", flush=True)

    # Also emit raw JSON for debugging
    print(json.dumps(obj), flush=True)


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


def env_reset(task_id: str) -> dict:
    import requests
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(rewritten_query: str) -> dict:
    import requests
    r = requests.post(f"{ENV_BASE_URL}/step", json={"rewritten_query": rewritten_query}, timeout=30)
    r.raise_for_status()
    return r.json()


def run_task(client, task_id: str) -> dict:
    env_available = True
    try:
        obs = env_reset(task_id)
    except Exception:
        obs = FALLBACK_OBSERVATIONS[task_id]
        env_available = False

    user_prompt = build_user_prompt(obs)
    start = time.time()

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
        return {
            "task_id":         task_id,
            "rewritten_query": rewritten_query,
            "reward":          {"total": 0.0, "correctness": 0.0, "efficiency": 0.0, "style": 0.0},
            "latency_s":       latency,
        }

    result = env_step(rewritten_query)
    return {
        "task_id":         task_id,
        "rewritten_query": rewritten_query,
        "reward":          result["reward"],
        "latency_s":       latency,
    }


def main():
    # ── [START] — emitted FIRST, before anything else can fail ───────────────
    _emit({
        "event":   "[START]",
        "model":   MODEL_NAME,
        "tasks":   TASK_IDS,
        "env_url": ENV_BASE_URL,
    })

    all_scores = []

    # ── Build OpenAI client pointed at the validator's LiteLLM proxy ─────────
    client = None
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
    except Exception as e:
        _emit({"event": "[ERROR]", "stage": "client_init", "error": str(e)})
        for i, task_id in enumerate(TASK_IDS, start=1):
            _emit({"event": "[STEP]", "step": i, "task_id": task_id, "score": 0.0,
                   "error": f"client init failed: {e}"})
            all_scores.append(0.0)
        _emit({
            "event":      "[END]",
            "scores":     {t: 0.0 for t in TASK_IDS},
            "mean_score": 0.0,
            "model":      MODEL_NAME,
        })
        return

    # ── Per-task loop ─────────────────────────────────────────────────────────
    for step_num, task_id in enumerate(TASK_IDS, start=1):
        try:
            result = run_task(client, task_id)
            score  = result["reward"]["total"]
            all_scores.append(score)
            rq = result["rewritten_query"]
            _emit({
                "event":           "[STEP]",
                "step":            step_num,
                "task_id":         task_id,
                "score":           score,
                "correctness":     result["reward"]["correctness"],
                "efficiency":      result["reward"]["efficiency"],
                "style":           result["reward"]["style"],
                "latency_s":       result["latency_s"],
                "rewritten_query": rq[:200] + "..." if len(rq) > 200 else rq,
            })
        except Exception as e:
            all_scores.append(0.0)
            _emit({"event": "[STEP]", "step": step_num, "task_id": task_id,
                   "score": 0.0, "error": str(e)})

    mean_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    # ── [END] — always emitted ────────────────────────────────────────────────
    _emit({
        "event":      "[END]",
        "scores":     {tid: s for tid, s in zip(TASK_IDS, all_scores)},
        "mean_score": mean_score,
        "model":      MODEL_NAME,
    })


if __name__ == "__main__":
    main()
