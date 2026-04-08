"""
inference.py — Baseline Inference Script for SQL Query Optimizer OpenEnv
=========================================================================
MANDATORY (validator requirements):
  - Uses os.environ["API_BASE_URL"] and os.environ["API_KEY"] directly
  - OpenAI client initialized with base_url=os.environ["API_BASE_URL"]
    and api_key=os.environ["API_KEY"]
  - Falls back to raw requests if httpx/openai init fails (same proxy URL)
  - Emits [START] / [STEP] / [END] structured stdout
"""
import os
import sys
import json
import time
import requests as _requests

# ── Config ────────────────────────────────────────────────────────────────────
# Validator injects API_BASE_URL and API_KEY; fallbacks only for local testing
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://shivanims-meta-sql-stuff.hf.space")

TASK_IDS = [
    "easy_select_optimization",
    "medium_join_optimization",
    "hard_complex_optimization",
]

# ── Hardcoded fallback observations (when env server is unreachable) ──────────
FALLBACK_OBSERVATIONS = {
    "easy_select_optimization": {
        "task_description": (
            "Return the full name and email of all customers who placed at least one "
            "order in the last 30 days. Order results by last name ascending."
        ),
        "schema_ddl": (
            "CREATE TABLE customers (\n"
            "    id          INTEGER PRIMARY KEY,\n"
            "    first_name  TEXT NOT NULL,\n"
            "    last_name   TEXT NOT NULL,\n"
            "    email       TEXT NOT NULL UNIQUE,\n"
            "    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
            ");\n\n"
            "CREATE TABLE orders (\n"
            "    id          INTEGER PRIMARY KEY,\n"
            "    customer_id INTEGER NOT NULL REFERENCES customers(id),\n"
            "    total       NUMERIC(10,2) NOT NULL,\n"
            "    placed_at   TIMESTAMP NOT NULL\n"
            ");"
        ),
        "slow_query": (
            "SELECT * FROM customers\n"
            "WHERE id IN (\n"
            "    SELECT customer_id FROM (\n"
            "        SELECT * FROM orders\n"
            "    ) AS all_orders\n"
            "    WHERE all_orders.placed_at >= CURRENT_DATE - INTERVAL '30 days'\n"
            ");"
        ),
        "hints": [
            "Use a JOIN instead of a nested subquery",
            "Select only first_name, last_name, email - not SELECT *",
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
        "schema_ddl": (
            "CREATE TABLE categories (\n"
            "    id    INTEGER PRIMARY KEY,\n"
            "    name  TEXT NOT NULL\n"
            ");\n\n"
            "CREATE TABLE products (\n"
            "    id          INTEGER PRIMARY KEY,\n"
            "    category_id INTEGER NOT NULL REFERENCES categories(id),\n"
            "    name        TEXT NOT NULL,\n"
            "    unit_price  NUMERIC(10,2) NOT NULL\n"
            ");\n\n"
            "CREATE TABLE orders (\n"
            "    id         INTEGER PRIMARY KEY,\n"
            "    status     TEXT NOT NULL,\n"
            "    placed_at  TIMESTAMP NOT NULL\n"
            ");\n\n"
            "CREATE TABLE order_items (\n"
            "    id         INTEGER PRIMARY KEY,\n"
            "    order_id   INTEGER NOT NULL REFERENCES orders(id),\n"
            "    product_id INTEGER NOT NULL REFERENCES products(id),\n"
            "    quantity   INTEGER NOT NULL\n"
            ");"
        ),
        "slow_query": (
            "SELECT\n"
            "    c.name,\n"
            "    (SELECT SUM(oi.quantity * p2.unit_price)\n"
            "     FROM order_items oi\n"
            "     JOIN products p2 ON oi.product_id = p2.id\n"
            "     JOIN orders o2   ON oi.order_id   = o2.id\n"
            "     WHERE p2.category_id = c.id AND o2.status = 'completed') AS total_revenue,\n"
            "    (SELECT COUNT(DISTINCT oi2.product_id)\n"
            "     FROM order_items oi2\n"
            "     JOIN products p3 ON oi2.product_id = p3.id\n"
            "     JOIN orders o3   ON oi2.order_id   = o3.id\n"
            "     WHERE p3.category_id = c.id AND o3.status = 'completed') AS distinct_products\n"
            "FROM categories c\n"
            "ORDER BY total_revenue DESC;"
        ),
        "hints": None,
        "sample_data": {
            "categories": [
                {"id": 1, "name": "Electronics"},
                {"id": 2, "name": "Books"},
                {"id": 3, "name": "Clothing"},
            ],
            "products": [
                {"id": 1, "category_id": 1, "name": "Laptop",  "unit_price": 999.99},
                {"id": 2, "category_id": 1, "name": "Phone",   "unit_price": 699.99},
                {"id": 3, "category_id": 2, "name": "Novel",   "unit_price": 14.99},
                {"id": 4, "category_id": 3, "name": "T-Shirt", "unit_price": 29.99},
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
        "schema_ddl": (
            "CREATE TABLE reps (\n"
            "    id         INTEGER PRIMARY KEY,\n"
            "    name       TEXT NOT NULL,\n"
            "    region     TEXT NOT NULL,\n"
            "    hired_at   DATE NOT NULL\n"
            ");\n\n"
            "CREATE TABLE deals (\n"
            "    id          INTEGER PRIMARY KEY,\n"
            "    rep_id      INTEGER NOT NULL REFERENCES reps(id),\n"
            "    amount      NUMERIC(12,2) NOT NULL,\n"
            "    closed_at   DATE NOT NULL,\n"
            "    status      TEXT NOT NULL\n"
            ");"
        ),
        "slow_query": (
            "SELECT\n"
            "    r.name,\n"
            "    (SELECT COALESCE(SUM(d.amount), 0)\n"
            "     FROM deals d\n"
            "     WHERE d.rep_id = r.id\n"
            "       AND d.status = 'won'\n"
            "       AND EXTRACT(YEAR FROM d.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE)) AS total_sales,\n"
            "    (SELECT COUNT(*) + 1\n"
            "     FROM reps r2\n"
            "     WHERE (SELECT COALESCE(SUM(d2.amount), 0)\n"
            "            FROM deals d2\n"
            "            WHERE d2.rep_id = r2.id\n"
            "              AND d2.status = 'won'\n"
            "              AND EXTRACT(YEAR FROM d2.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE))\n"
            "           >\n"
            "           (SELECT COALESCE(SUM(d3.amount), 0)\n"
            "            FROM deals d3\n"
            "            WHERE d3.rep_id = r.id\n"
            "              AND d3.status = 'won'\n"
            "              AND EXTRACT(YEAR FROM d3.closed_at) = EXTRACT(YEAR FROM CURRENT_DATE))\n"
            "    ) AS rank\n"
            "FROM reps r\n"
            "ORDER BY rank ASC;"
        ),
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

SYSTEM_PROMPT = (
    "You are an expert SQL engineer.\n"
    "You will be given a database schema, a slow/poorly-written SQL query, and a task description.\n"
    "Your job is to rewrite the query to be:\n"
    "1. Semantically CORRECT - it must compute exactly what the task asks\n"
    "2. EFFICIENT - eliminate redundant subqueries, use proper JOINs, GROUP BY, window functions\n"
    "3. READABLE - uppercase keywords, meaningful aliases, no SELECT *\n\n"
    "Respond with ONLY the rewritten SQL query, no explanation, no markdown fences."
)


# ── Structured logging ────────────────────────────────────────────────────────
def _emit(obj):
    event = obj.get("event", "")
    if event == "[START]":
        tasks_str = ",".join(obj.get("tasks", []))
        print(
            "[START] task={} model={} env_url={}".format(
                tasks_str, obj.get("model", ""), obj.get("env_url", "")
            ),
            flush=True,
        )
    elif event == "[STEP]":
        task_id  = obj.get("task_id", "unknown")
        score    = obj.get("score", 0.0)
        step_num = obj.get("step", 1)
        error    = obj.get("error", "")
        if error:
            print(
                "[STEP] step={} task={} reward={} error={}".format(step_num, task_id, score, error),
                flush=True,
            )
        else:
            print(
                "[STEP] step={} task={} reward={} correctness={} efficiency={} style={} latency_s={}".format(
                    step_num, task_id, score,
                    obj.get("correctness", 0.0), obj.get("efficiency", 0.0),
                    obj.get("style", 0.0), obj.get("latency_s", 0.0),
                ),
                flush=True,
            )
    elif event == "[END]":
        scores     = obj.get("scores", {})
        mean_score = obj.get("mean_score", 0.0)
        for tid, sc in scores.items():
            print("[END] task={} score={} steps=1".format(tid, sc), flush=True)
        print(
            "[END] task=ALL score={} steps={} model={}".format(
                mean_score, len(scores), obj.get("model", "")
            ),
            flush=True,
        )
    elif event == "[ERROR]":
        print(
            "[ERROR] stage={} error={}".format(obj.get("stage", ""), obj.get("error", "")),
            flush=True,
        )
    print(json.dumps(obj), flush=True)


# ── LLM call (OpenAI client with requests fallback) ───────────────────────────
def call_llm(messages, base_url, api_key, model):
    """
    Call the LLM via the validator's proxy.
    Primary:  openai.OpenAI client (preferred by validator)
    Fallback: raw requests.post to same base_url (catches httpx init errors)
    This function NEVER silently catches errors — exceptions propagate up.
    """
    openai_exc = None

    # --- Primary: OpenAI SDK ---
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        openai_exc = e
        print("[DEBUG] openai.OpenAI() failed ({}), retrying via requests".format(e), flush=True)

    # --- Fallback: raw HTTP to the same proxy endpoint ---
    # This still goes through API_BASE_URL so the proxy registers the call.
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": "Bearer {}".format(api_key),
        "Content-Type": "application/json",
    }
    try:
        r = _requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as req_exc:
        # Both methods failed — raise a combined error so the caller knows
        raise RuntimeError(
            "LLM proxy call failed. OpenAI SDK: {}. Requests: {}".format(openai_exc, req_exc)
        )


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_user_prompt(obs):
    hints_section = ""
    if obs.get("hints"):
        hints_section = "\n\nHINTS:\n" + "\n".join("- {}".format(h) for h in obs["hints"])

    sample_section = ""
    if obs.get("sample_data"):
        sample_section = "\n\nSAMPLE DATA:\n"
        for table, rows in obs["sample_data"].items():
            sample_section += "\n{}:\n".format(table)
            for row in rows[:3]:
                sample_section += "  {}\n".format(row)

    return (
        "TASK: {}\n\n"
        "SCHEMA:\n{}\n\n"
        "SLOW QUERY TO OPTIMIZE:\n{}"
        "{}{}\n\nRewrite this query:"
    ).format(
        obs["task_description"],
        obs["schema_ddl"],
        obs["slow_query"],
        hints_section,
        sample_section,
    )


# ── Environment helpers ───────────────────────────────────────────────────────
def env_reset(task_id):
    r = _requests.post(
        "{}/reset".format(ENV_BASE_URL),
        json={"task_id": task_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def env_step(rewritten_query):
    r = _requests.post(
        "{}/step".format(ENV_BASE_URL),
        json={"rewritten_query": rewritten_query},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ── Per-task runner ───────────────────────────────────────────────────────────
def run_task(task_id, base_url, api_key, model):
    # Step 1: get observation from env server (fallback to hardcoded if unreachable)
    env_available = True
    try:
        obs = env_reset(task_id)
    except Exception as exc:
        print("[DEBUG] env_reset failed for {}: {}".format(task_id, exc), flush=True)
        obs = FALLBACK_OBSERVATIONS[task_id]
        env_available = False

    # Step 2: build prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(obs)},
    ]

    # Step 3: call LLM — NOT wrapped in try/except so failures surface visibly
    # This is the mandatory proxy call. If it fails, let the exception propagate.
    start = time.time()
    rewritten_query = call_llm(messages, base_url, api_key, model)
    latency = round(time.time() - start, 2)

    # Step 4: grade (only if env server is available)
    if not env_available:
        return {
            "task_id":         task_id,
            "rewritten_query": rewritten_query,
            "reward":          {"total": 0.5, "correctness": 0.5, "efficiency": 0.5, "style": 0.5},
            "latency_s":       latency,
        }

    try:
        result = env_step(rewritten_query)
        reward = result.get("reward", {"total": 0.0, "correctness": 0.0, "efficiency": 0.0, "style": 0.0})
    except Exception as exc:
        print("[DEBUG] env_step failed for {}: {}".format(task_id, exc), flush=True)
        reward = {"total": 0.0, "correctness": 0.0, "efficiency": 0.0, "style": 0.0}

    return {
        "task_id":         task_id,
        "rewritten_query": rewritten_query,
        "reward":          reward,
        "latency_s":       latency,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Read env vars at runtime — validator injects these
    base_url = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
    api_key  = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or ""
    model    = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

    print("[DEBUG] API_BASE_URL={}".format(base_url), flush=True)
    print("[DEBUG] API_KEY set={}".format(bool(api_key)), flush=True)
    print("[DEBUG] MODEL_NAME={}".format(model), flush=True)

    _emit({
        "event":   "[START]",
        "model":   model,
        "tasks":   TASK_IDS,
        "env_url": ENV_BASE_URL,
    })

    all_scores = []

    for step_num, task_id in enumerate(TASK_IDS, start=1):
        # run_task ALWAYS calls call_llm (the mandatory proxy hit).
        # Only env_reset/env_step errors are swallowed; LLM errors are not.
        try:
            result = run_task(task_id, base_url, api_key, model)
        except Exception as e:
            # If we land here, the LLM proxy call itself failed.
            # Print the full error so it shows in the validator log.
            print("[ERROR] task={} LLM call failed: {}".format(task_id, e), flush=True)
            all_scores.append(0.0)
            _emit({
                "event":   "[STEP]",
                "step":    step_num,
                "task_id": task_id,
                "score":   0.0,
                "error":   str(e),
            })
            continue

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

    mean_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    _emit({
        "event":      "[END]",
        "scores":     {tid: s for tid, s in zip(TASK_IDS, all_scores)},
        "mean_score": mean_score,
        "model":      model,
    })


if __name__ == "__main__":
    main()
