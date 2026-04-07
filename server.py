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


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Env running 🚀"}
@app.get("/health")
def health():
    return {"status": "ok", "env": "sql-query-optimizer", "version": "1.0.0"}


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


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
