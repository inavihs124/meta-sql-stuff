---
title: SQL Query Optimizer OpenEnv
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - rl-environment
  - sql
  - code-optimization
  - reinforcement-learning
---

# SQL Query Optimizer — OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://openenv.dev)

An RL environment where agents optimize slow SQL queries into fast, readable, correct ones.

## Overview

Given a **database schema** and a **slow, poorly-written SQL query**, the agent must rewrite it to be:
- ✅ **Correct** — semantically equivalent to the task description
- ⚡ **Efficient** — eliminates N+1 subqueries, uses proper JOINs and window functions
- 📖 **Readable** — clean SQL style, no `SELECT *`, proper aliases

This fills a real gap: SQL rewriting / query optimization is a genuine daily task for data engineers, DBAs, and analysts. Training agents on this enables automated query review tools, IDE plugins, and database tutors.

---

## Action & Observation Space

### Observation
| Field | Type | Description |
|---|---|---|
| `schema_ddl` | `str` | `CREATE TABLE` DDL for all tables |
| `slow_query` | `str` | The original slow/bad SQL to optimize |
| `task_description` | `str` | Natural language spec of what the query must compute |
| `sample_data` | `dict` | Sample rows per table for context |
| `hints` | `list[str] \| None` | Optional hints (easy task only) |
| `task_id` | `str` | Current task identifier |

### Action
| Field | Type | Description |
|---|---|---|
| `rewritten_query` | `str` | The agent's optimized SQL query |

### Reward
| Field | Type | Description |
|---|---|---|
| `total` | `float [0,1]` | Weighted composite score |
| `correctness` | `float [0,1]` | Semantic correctness (weight 0.5) |
| `efficiency` | `float [0,1]` | Query efficiency (weight 0.3) |
| `style` | `float [0,1]` | SQL style/readability (weight 0.2) |
| `breakdown` | `dict` | Detailed sub-scores and detected issues |

---

## Tasks

### 🟢 Easy — Basic SELECT Optimization
Find customers with recent orders. Slow query uses `SELECT *` and a pointless double-nested subquery.
**Expected score for a decent agent: 0.75–0.95**

### 🟡 Medium — JOIN & Aggregation Optimization  
Category revenue report. Slow query uses correlated subqueries for each aggregate instead of a single GROUP BY.
**Expected score for a decent agent: 0.60–0.85**

### 🔴 Hard — Window Functions & CTE  
Sales rep leaderboard with ranking and percentage. Slow query uses a triple-nested correlated subquery for ranking instead of `RANK() OVER (...)`.
**Expected score for a decent agent: 0.45–0.75**

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks |
| `POST` | `/reset` | Reset environment, returns Observation |
| `POST` | `/step` | Submit action, returns StepResult |
| `GET` | `/state` | Current environment state |

### Quick example
```bash
# Reset to easy task
curl -X POST https://your-space.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "easy_select_optimization"}'

# Submit optimized query
curl -X POST https://your-space.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{"rewritten_query": "SELECT DISTINCT c.first_name, c.last_name, c.email FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.placed_at >= CURRENT_DATE - INTERVAL '\''30 days'\'' ORDER BY c.last_name ASC"}'
```

---

## Setup & Usage

### Local with Docker
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/sql-query-optimizer
cd sql-query-optimizer
docker build -t sql-opt-env .
docker run -p 7860:7860 sql-opt-env
```

### Local with Python
```bash
pip install -r requirements.txt
python server.py
```

### Run Baseline Inference
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
export HF_TOKEN=hf_your_token
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

---

## Baseline Scores (gpt-4o, temperature=0)

| Task | Score |
|---|---|
| easy_select_optimization | 0.89 |
| medium_join_optimization | 0.74 |
| hard_complex_optimization | 0.61 |
| **Mean** | **0.75** |

---

## Reward Design

The reward function is **trajectory-aware** and provides **partial progress signals**:

- **Correctness (50%)**: Presence of required SQL clauses, absence of known anti-patterns
- **Efficiency (30%)**: Correlated subquery detection, CTE/window function usage, nesting depth
- **Style (20%)**: No `SELECT *`, meaningful aliases, uppercase keywords, clean formatting

Rewards are fully deterministic — no LLM in the grader loop, no DB execution needed.

---

## File Structure
```
sql-query-optimizer/
├── server.py              # FastAPI HTTP server
├── inference.py           # Baseline inference script (required)
├── openenv.yaml           # OpenEnv metadata
├── requirements.txt
├── Dockerfile
├── README.md
├── env/
│   ├── __init__.py
│   ├── models.py          # Pydantic Observation/Action/Reward models
│   └── environment.py     # SQLOptimizerEnv class
└── tasks/
    ├── __init__.py
    ├── task_definitions.py  # 3 tasks with schemas & slow queries
    └── graders.py           # Deterministic graders
```
