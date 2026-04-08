"""
OpenEnv HTTP Server — FastAPI
Exposes /reset, /step, /state, /tasks, /health
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

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

# Single shared environment instance (stateful, single-user for HF Space)
_env = SQLOptimizerEnv()


# ── Request / Response bodies ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    rewritten_query: str


# ── NEW: MCP request body ─────────────────────────────────────────────────────
class MCPRequest(BaseModel):
    jsonrpc: Optional[str] = "2.0"
    method: Optional[str] = None
    params: Optional[dict] = None
    id: Optional[int] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Env running 🚀"}


@app.get("/health")
def health():
    return {"status": "healthy", "env": "sql-query-optimizer", "version": "1.0.1"}


# ── NEW: /metadata ────────────────────────────────────────────────────────────
@app.get("/metadata")
def metadata():
    return {
        "name": "sql-query-optimizer",
        "description": (
            "An RL environment where agents optimize slow SQL queries. "
            "Given a schema and a poorly-written query, the agent rewrites it "
            "to be correct, efficient, and readable."
        ),
        "version": "1.0.1",
        "tags": ["sql", "code-optimization", "real-world", "openenv"],
    }


# ── NEW: /schema ──────────────────────────────────────────────────────────────
@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "rewritten_query": {"type": "string", "description": "The agent's rewritten SQL query"}
            },
            "required": ["rewritten_query"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "schema_ddl":       {"type": "string"},
                "slow_query":       {"type": "string"},
                "task_description": {"type": "string"},
                "sample_data":      {"type": "object"},
                "hints":            {"type": "array", "items": {"type": "string"}, "nullable": True},
                "task_id":          {"type": "string"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id":      {"type": "string"},
                "step_count":   {"type": "integer"},
                "last_action":  {"nullable": True},
                "last_reward":  {"nullable": True},
                "episode_done": {"type": "boolean"},
            },
        },
    }


# ── NEW: /mcp (JSON-RPC 2.0) ──────────────────────────────────────────────────
@app.post("/mcp")
def mcp(body: MCPRequest = MCPRequest()):
    method = body.method or "ping"

    if method == "ping":
        return {"jsonrpc": "2.0", "result": {"status": "ok", "env": "sql-query-optimizer"}, "id": body.id}

    if method == "reset":
        params = body.params or {}
        obs = _env.reset(task_id=params.get("task_id"))
        return {"jsonrpc": "2.0", "result": obs.model_dump(), "id": body.id}

    if method == "step":
        params = body.params or {}
        action = Action(rewritten_query=params.get("rewritten_query", ""))
        result = _env.step(action)
        return {"jsonrpc": "2.0", "result": result.model_dump(), "id": body.id}

    if method == "state":
        return {"jsonrpc": "2.0", "result": _env.state().model_dump(), "id": body.id}

    return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method not found: {method}"}, "id": body.id}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id":          tid,
                "name":        TASKS[tid]["task_description"][:60] + "...",
                "difficulty":  TASKS[tid]["difficulty"],
                "has_grader":  True,
                "grader":      True,
                "grader_type": "programmatic",
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


# ── NEW: main() + __main__ block (required by openenv validate) ───────────────
def main():
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)  # UNCHANGED module path


if __name__ == "__main__":
    main()
