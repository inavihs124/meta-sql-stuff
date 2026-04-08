"""
inference.py — Baseline Inference Script for SQL Query Optimizer OpenEnv
STRICTLY uses validator-provided API_BASE_URL and API_KEY
"""

import os
import json
import time

# ── Config ─────────────────────────────────────────────────────────────
API_BASE_URL = os.environ["API_BASE_URL"]   # MUST use validator value
API_KEY      = os.environ["API_KEY"]        # MUST use validator value
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASK_IDS = [
    "easy_select_optimization",
    "medium_join_optimization",
    "hard_complex_optimization",
]

SYSTEM_PROMPT = """You are an expert SQL engineer.
Rewrite the given SQL query to be:
1. Correct
2. Efficient
3. Readable

Respond ONLY with SQL. No explanation.
"""


# ── Output Formatter (REQUIRED FORMAT) ─────────────────────────────────
def _emit(obj: dict):
    event = obj.get("event", "")

    if event == "[START]":
        tasks_str = ",".join(obj.get("tasks", []))
        print(f"[START] task={tasks_str} model={obj.get('model')} env_url={obj.get('env_url')}", flush=True)

    elif event == "[STEP]":
        if obj.get("error"):
            print(f"[STEP] step={obj['step']} task={obj['task_id']} reward=0 error={obj['error']}", flush=True)
        else:
            print(
                f"[STEP] step={obj['step']} task={obj['task_id']} reward={obj['score']} "
                f"correctness={obj['correctness']} efficiency={obj['efficiency']} "
                f"style={obj['style']} latency_s={obj['latency_s']}",
                flush=True,
            )

    elif event == "[END]":
        for task_id, score in obj["scores"].items():
            print(f"[END] task={task_id} score={score} steps=1", flush=True)
        print(f"[END] task=ALL score={obj['mean_score']} steps={len(obj['scores'])}", flush=True)

    print(json.dumps(obj), flush=True)


# ── Prompt Builder ─────────────────────────────────────────────────────
def build_prompt(obs):
    return f"""
TASK:
{obs['task_description']}

SCHEMA:
{obs['schema_ddl']}

SLOW QUERY:
{obs['slow_query']}

Rewrite this query:
"""


# ── Environment Calls ──────────────────────────────────────────────────
def env_reset(task_id):
    import requests
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(query):
    import requests
    r = requests.post(f"{ENV_BASE_URL}/step", json={"rewritten_query": query}, timeout=30)
    r.raise_for_status()
    return r.json()


# ── Main Execution ─────────────────────────────────────────────────────
def main():
    from openai import OpenAI

    # ✅ STRICTLY use validator proxy
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    _emit({
        "event": "[START]",
        "model": MODEL_NAME,
        "tasks": TASK_IDS,
        "env_url": ENV_BASE_URL,
    })

    scores = []

    for step_num, task_id in enumerate(TASK_IDS, start=1):
        try:
            obs = env_reset(task_id)
            prompt = build_prompt(obs)

            start = time.time()

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )

            latency = round(time.time() - start, 2)

            query = response.choices[0].message.content.strip()

            result = env_step(query)
            reward = result["reward"]

            score = reward["total"]
            scores.append(score)

            _emit({
                "event": "[STEP]",
                "step": step_num,
                "task_id": task_id,
                "score": score,
                "correctness": reward["correctness"],
                "efficiency": reward["efficiency"],
                "style": reward["style"],
                "latency_s": latency,
            })

        except Exception as e:
            scores.append(0.0)
            _emit({
                "event": "[STEP]",
                "step": step_num,
                "task_id": task_id,
                "score": 0.0,
                "error": str(e),
            })

    mean_score = round(sum(scores) / len(scores), 4)

    _emit({
        "event": "[END]",
        "scores": dict(zip(TASK_IDS, scores)),
        "mean_score": mean_score,
    })


if __name__ == "__main__":
    main()