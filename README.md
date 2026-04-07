# SQL Query Optimizer — OpenEnv Environment

## Overview

This project is an OpenEnv-compatible reinforcement learning environment where an agent learns to rewrite slow, poorly written SQL queries into optimized, readable, and correct ones. The idea came from a real frustration — writing SQL that *works* is easy, but writing SQL that is actually *good* is a completely different challenge. Engineers spend a lot of time fixing inefficient queries, removing unnecessary nesting, and cleaning up code so that teammates can understand it. This environment simulates exactly that workflow.

We built this to be useful for:

* Evaluating how well LLMs reason about SQL
* Training optimization agents using a dense reward signal
* Building tools like SQL assistants or automated code review systems

---

## What the Agent Does

The agent receives:

* A **database schema** (CREATE TABLE statements)
* A **slow or poorly written SQL query**
* A **task description** explaining what the query is supposed to do
* Some **sample data** for context
* Optional **hints** (only for easy tasks)

Its job is to rewrite the query so that it is:

* ✅ **Correct** — produces the expected output
* ⚡ **Efficient** — avoids patterns like nested subqueries and redundant operations
* 📖 **Readable** — clean, well-structured SQL that another engineer can easily understand

---

## Observation & Action Space

**Input (Observation):**

| Field              | Description                      |
| ------------------ | -------------------------------- |
| `schema_ddl`       | Table definitions                |
| `slow_query`       | The inefficient query to improve |
| `task_description` | What the query should do         |
| `sample_data`      | Example rows for context         |
| `hints`            | Optional hints for easier tasks  |
| `task_id`          | Unique identifier for the task   |

**Output (Action):**

| Field             | Description                    |
| ----------------- | ------------------------------ |
| `rewritten_query` | The agent’s improved SQL query |

---

## Reward Design

Rather than using a simple pass/fail signal, the environment provides **partial credit** across three dimensions:

| Component   | Weight | What It Checks                                                           |
| ----------- | ------ | ------------------------------------------------------------------------ |
| Correctness | 0.5    | Does the output match the expected result?                               |
| Efficiency  | 0.3    | Are inefficient patterns (like nested subqueries or redundancy) avoided? |
| Style       | 0.2    | Is the SQL clean, readable, and well-structured?                         |

This creates a **dense reward signal**, allowing agents to improve gradually instead of only receiving feedback when everything is perfectly correct.

---

## Task Difficulty Levels

### 🟢 Easy — Basic Cleanup

* **Problem:** `SELECT *` with unnecessary nesting
* **Goal:** Simplify and clean the query
* **Expected Score:** 0.75 – 0.95

### 🟡 Medium — Aggregation Fix

* **Problem:** Repeated correlated subqueries performing aggregation
* **Goal:** Rewrite using `GROUP BY` efficiently
* **Expected Score:** 0.60 – 0.85

### 🔴 Hard — Advanced Optimization

* **Problem:** Deep nested queries used for ranking
* **Goal:** Replace them with window functions like `RANK() OVER (...)`
* **Expected Score:** 0.45 – 0.75

---

## API Endpoints

| Method | Endpoint  | Description                        |
| ------ | --------- | ---------------------------------- |
| GET    | `/health` | Check if the server is running     |
| GET    | `/tasks`  | List all available tasks           |
| POST   | `/reset`  | Start a new task                   |
| POST   | `/step`   | Submit a rewritten query           |
| GET    | `/state`  | View the current environment state |

---

## Quick Example

```bash
# Start a task
curl -X POST https://your-space.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_select_optimization"}'

# Submit your solution
curl -X POST https://your-space.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"rewritten_query": "SELECT DISTINCT c.first_name, c.last_name, c.email FROM customers c JOIN orders o ON c.id = o.customer_id"}'
```

---

## How to Run

### With Docker (Recommended)

```bash
docker build -t sql-opt-env .
docker run -p 7860:7860 sql-opt-env
```

---

### Local Python

```bash
pip install -r requirements.txt
python server.py
```

---

### Run the Baseline Agent

```bash
export OPENAI_API_KEY=your_key
python inference.py
```

---

## Baseline Performance

| Task        | Score    |
| ----------- | -------- |
| Easy        | 0.89     |
| Medium      | 0.74     |
| Hard        | 0.61     |
| **Average** | **0.75** |

---

## Project Structure

```
sql-query-optimizer/
├── server.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── README.md
├── env/
│   ├── models.py
│   └── environment.py
└── tasks/
    ├── task_definitions.py
    └── graders.py
```

