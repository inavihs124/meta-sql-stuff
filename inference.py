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
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TASK_IDS = [
    "easy_select_optimization",
    "medium_join_optimization",
    "hard_complex_optimization",
]

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
