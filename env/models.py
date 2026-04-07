"""
OpenEnv SQL Query Optimizer — Typed Pydantic Models
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class Observation(BaseModel):
    schema_ddl: str = Field(..., description="Database schema as CREATE TABLE DDL statements")
    slow_query: str = Field(..., description="The original slow/poorly-written SQL query")
    task_description: str = Field(..., description="Natural language description of what the query must compute")
    sample_data: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Sample rows per table for context: {table_name: [row_dict, ...]}"
    )
    hints: Optional[List[str]] = Field(
        default=None,
        description="Optional hints provided on easy tasks"
    )
    task_id: str = Field(..., description="Current task identifier")


class Action(BaseModel):
    rewritten_query: str = Field(
        ...,
        description="The agent's rewritten/optimized SQL query"
    )


class Reward(BaseModel):
    total: float = Field(..., ge=0.0, le=1.0, description="Total reward [0.0, 1.0]")
    correctness: float = Field(..., ge=0.0, le=1.0, description="Semantic correctness score")
    efficiency: float = Field(..., ge=0.0, le=1.0, description="Query efficiency / optimization score")
    style: float = Field(..., ge=0.0, le=1.0, description="SQL style and readability score")
    breakdown: Dict[str, Any] = Field(default_factory=dict, description="Detailed scoring breakdown")


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StateSnapshot(BaseModel):
    task_id: str
    step_count: int
    last_action: Optional[Action]
    last_reward: Optional[Reward]
    episode_done: bool
