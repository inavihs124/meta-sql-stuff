"""
SQL Query Optimizer — OpenEnv Environment
Implements: step(), reset(), state()
"""
from typing import Optional
from env.models import Observation, Action, Reward, StepResult, StateSnapshot
from tasks.task_definitions import TASKS
from tasks.graders import grade

TASK_ORDER = [
    "easy_select_optimization",
    "medium_join_optimization",
    "hard_complex_optimization",
]


class SQLOptimizerEnv:
    """
    OpenEnv-compliant SQL Query Optimization environment.

    The agent receives a schema + slow query + task description and must
    return an optimized SQL query. Graders evaluate correctness, efficiency,
    and style without executing the SQL (fully deterministic, no DB needed).
    """

    def __init__(self, task_id: Optional[str] = None):
        self._task_id: str = task_id or TASK_ORDER[0]
        self._step_count: int = 0
        self._last_action: Optional[Action] = None
        self._last_reward: Optional[Reward] = None
        self._done: bool = False

        if self._task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{self._task_id}'. "
                f"Valid options: {list(TASKS.keys())}"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Reset the environment and return the initial observation."""
        if task_id:
            if task_id not in TASKS:
                raise ValueError(f"Unknown task_id: {task_id}")
            self._task_id = task_id

        self._step_count = 0
        self._last_action = None
        self._last_reward = None
        self._done = False

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Execute one step: grade the agent's rewritten query.
        Returns (observation, reward, done=True, info).
        Each episode is exactly one step.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        task_def = TASKS[self._task_id]
        reward = grade(self._task_id, task_def, action)

        self._step_count += 1
        self._last_action = action
        self._last_reward = reward
        self._done = True  # single-step episodes

        obs = self._build_observation()

        return StepResult(
            observation=obs,
            reward=reward,
            done=True,
            info={
                "task_id": self._task_id,
                "difficulty": task_def["difficulty"],
                "step": self._step_count,
            }
        )

    def state(self) -> StateSnapshot:
        """Return a snapshot of the current environment state."""
        return StateSnapshot(
            task_id=self._task_id,
            step_count=self._step_count,
            last_action=self._last_action,
            last_reward=self._last_reward,
            episode_done=self._done,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        task_def = TASKS[self._task_id]
        return Observation(
            schema_ddl=task_def["schema_ddl"].strip(),
            slow_query=task_def["slow_query"].strip(),
            task_description=task_def["task_description"],
            sample_data=task_def.get("sample_data", {}),
            hints=task_def.get("hints"),
            task_id=self._task_id,
        )
