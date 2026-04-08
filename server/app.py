"""
OpenEnv HTTP Server — FastAPI
Exposes all required OpenEnv endpoints:
  /health, /metadata, /schema, /mcp, /reset, /step, /state, /tasks
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import from parent package (works when run from project root)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import SQLOptimizerEnv, TASK_ORDER
from env.models import Action, Observation, Reward, StepResult, StateSnapshot
from tasks.task_definitions import TASKS

app = FastAPI(
    title="SQL Query Optimizer — OpenEnv",
    description="RL environment for optimizing SQL queries. Implements OpenEnv step/reset/state API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance
_env = SQLOptimizerEnv()


# ── Request / Response bodies ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    rewritten_query: str


class MCPRequest(BaseModel):
    jsonrpc: Optional[str] = "2.0"
    method: Optional[str] = None
    params: Optional[dict] = None
    id: Optional[int] = None


# ── Required OpenEnv endpoints ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "SQL Query Optimizer OpenEnv 🚀", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "env": "sql-query-optimizer", "version": "1.0.0"}


@app.get("/metadata")
def metadata():
    """OpenEnv required: returns environment name and description."""
    return {
        "name": "sql-query-optimizer",
        "description": (
            "An RL environment where agents optimize slow SQL queries. "
            "Given a schema and a poorly-written query, the agent rewrites it "
            "to be correct, efficient, and readable."
        ),
        "version": "1.0.0",
        "author": "openenv-submission",
        "tags": ["sql", "code-optimization", "real-world", "openenv"],
        "tasks": [
            {
                "id": tid,
                "difficulty": TASKS[tid]["difficulty"],
                "description": TASKS[tid]["task_description"][:80],
            }
            for tid in TASK_ORDER
        ],
    }


@app.get("/schema")
def schema():
    """OpenEnv required: returns action, observation, and state schemas."""
    return {
        "action": {
            "type": "object",
            "description": "A rewritten/optimized SQL query",
            "properties": {
                "rewritten_query": {
                    "type": "string",
                    "description": "The agent's rewritten SQL query",
                }
            },
            "required": ["rewritten_query"],
        },
        "observation": {
            "type": "object",
            "description": "Environment observation: schema + slow query + task context",
            "properties": {
                "schema_ddl": {"type": "string", "description": "Database schema DDL"},
                "slow_query": {"type": "string", "description": "Original slow SQL query"},
                "task_description": {"type": "string", "description": "What the query must compute"},
                "sample_data": {"type": "object", "description": "Sample rows per table"},
                "hints": {"type": "array", "items": {"type": "string"}, "nullable": True},
                "task_id": {"type": "string", "description": "Current task identifier"},
            },
        },
        "state": {
            "type": "object",
            "description": "Current environment state snapshot",
            "properties": {
                "task_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "last_action": {"nullable": True},
                "last_reward": {"nullable": True},
                "episode_done": {"type": "boolean"},
            },
        },
    }


@app.post("/mcp")
def mcp(body: MCPRequest = MCPRequest()):
    """OpenEnv required: JSON-RPC 2.0 compatible MCP endpoint."""
    method = body.method or "ping"

    if method == "ping" or method is None:
        return {
            "jsonrpc": "2.0",
            "result": {"status": "ok", "env": "sql-query-optimizer"},
            "id": body.id,
        }

    if method == "reset":
        params = body.params or {}
        obs = _env.reset(task_id=params.get("task_id"))
        return {
            "jsonrpc": "2.0",
            "result": obs.model_dump(),
            "id": body.id,
        }

    if method == "step":
        params = body.params or {}
        action = Action(rewritten_query=params.get("rewritten_query", ""))
        result = _env.step(action)
        return {
            "jsonrpc": "2.0",
            "result": result.model_dump(),
            "id": body.id,
        }

    if method == "state":
        return {
            "jsonrpc": "2.0",
            "result": _env.state().model_dump(),
            "id": body.id,
        }

    return {
        "jsonrpc": "2.0",
        "error": {"code": -32601, "message": f"Method not found: {method}"},
        "id": body.id,
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": tid,
                "name": TASKS[tid]["task_description"][:60] + "...",
                "difficulty": TASKS[tid]["difficulty"],
            }
            for tid in TASK_ORDER
        ]
    }


@app.post("/reset", response_model=Observation)
def reset(body: ResetRequest = ResetRequest()):
    try:
        obs = _env.reset(task_id=body.task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(body: StepRequest):
    try:
        action = Action(rewritten_query=body.rewritten_query)
        result = _env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=StateSnapshot)
def state():
    return _env.state()


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
