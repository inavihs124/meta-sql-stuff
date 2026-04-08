"""
inference.py — Baseline Inference Script for SQL Query Optimizer OpenEnv
Reads: OPENAI_API_KEY, API_BASE_URL, MODEL_NAME, ENV_BASE_URL
Emits structured stdout: [START], [STEP], [END] format (required by OpenEnv bootcamp)

Usage:
    OPENAI_API_KEY=sk-... MODEL_NAME=gpt-4o python inference.py
"""
import os
import sys
import json
import time
import requests

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")   # FIXED: was reading HF_TOKEN

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASK_IDS = [
    "easy_select_optimization",
    "medium_join_optimization",
    "hard_complex_optimization",
]

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(rewritten_query: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json={"rewritten_query": rewritten_query}, timeout=30)
    r.raise_for_status()
    return r.json()

# ── Agent prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert SQL engineer.
You will be given a database schema, a slow/poorly-written SQL query, and a task description.
Your job is to rewrite the query to be:
1. Semantically CORRECT — it must compute exactly what the task asks
2. EFFICIENT — eliminate redundant subqueries, use proper JOINs, GROUP BY, window functions
3. READABLE — uppercase keywords, meaningful aliases, no SELECT *

Respond with ONLY the rewritten SQL query, no explanation, no markdown fences."""

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

    return f"""TASK: {obs['task_description']}

SCHEMA:
{obs['schema_ddl']}

SLOW QUERY TO OPTIMIZE:
{obs['slow_query']}{hints_section}{sample_section}

Rewrite this query:"""


# ── Main inference loop ───────────────────────────────────────────────────────

def run_task(client, task_id: str) -> dict:
    obs = env_reset(task_id)
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

    result = env_step(rewritten_query)
    reward = result["reward"]

    return {
        "task_id": task_id,
        "rewritten_query": rewritten_query,
        "reward": reward,
        "latency_s": latency,
    }


def main():
    all_scores = []

    # ── [START] ──────────────────────────────────────────────────────────────
    print(json.dumps({
        "event": "[START]",
        "model": MODEL_NAME,
        "tasks": TASK_IDS,
        "env_url": ENV_BASE_URL,
    }))
    sys.stdout.flush()

    # FIXED: OpenAI client created inside main(), inside try/except
    # — never at module level, so import alone can't crash the validator
    try:
        from openai import OpenAI
        api_key = OPENAI_API_KEY or "dummy-key-not-set"
        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    except Exception as e:
        print(json.dumps({"event": "[ERROR]", "stage": "client_init", "error": str(e)}))
        sys.stdout.flush()
        # Emit zero scores and exit cleanly — no unhandled exception
        for task_id in TASK_IDS:
            print(json.dumps({"event": "[STEP]", "task_id": task_id, "score": 0.0, "error": "client init failed"}))
        print(json.dumps({"event": "[END]", "scores": {t: 0.0 for t in TASK_IDS}, "mean_score": 0.0, "model": MODEL_NAME}))
        sys.stdout.flush()
        sys.exit(0)   # exit 0 so the validator sees a clean process exit

    for task_id in TASK_IDS:
        try:
            result = run_task(client, task_id)
            score = result["reward"]["total"]
            all_scores.append(score)

            # ── [STEP] ───────────────────────────────────────────────────────
            print(json.dumps({
                "event": "[STEP]",
                "task_id": task_id,
                "score": score,
                "correctness": result["reward"]["correctness"],
                "efficiency":  result["reward"]["efficiency"],
                "style":       result["reward"]["style"],
                "latency_s":   result["latency_s"],
                "rewritten_query": result["rewritten_query"][:200] + "..." if len(result["rewritten_query"]) > 200 else result["rewritten_query"],
            }))
            sys.stdout.flush()

        except Exception as e:
            print(json.dumps({
                "event": "[STEP]",
                "task_id": task_id,
                "score": 0.0,
                "error": str(e),
            }))
            sys.stdout.flush()
            all_scores.append(0.0)

    mean_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    # ── [END] ────────────────────────────────────────────────────────────────
    print(json.dumps({
        "event": "[END]",
        "scores": {tid: s for tid, s in zip(TASK_IDS, all_scores)},
        "mean_score": mean_score,
        "model": MODEL_NAME,
    }))
    sys.stdout.flush()


if __name__ == "__main__":
    main()