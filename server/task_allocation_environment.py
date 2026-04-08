"""
Multi-Agent Task Allocation Environment.

Three tasks of increasing difficulty:
  0  easy_allocation    (easy)    assign 3 tasks with clear skill matches
  1  medium_allocation  (medium)  allocate 7 mixed tasks efficiently
  2  hard_allocation    (hard)    optimize 12 tasks across overloaded team
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TaskAllocationAction, TaskAllocationObservation
except ImportError:
    try:
        from models import TaskAllocationAction, TaskAllocationObservation
    except ImportError:
        from multi_agent_task_alloc.models import TaskAllocationAction, TaskAllocationObservation


TASK_NAMES = ["easy_allocation", "medium_allocation", "hard_allocation"]

TEAM = [
    {"id": "alice",  "name": "Alice",  "skills": ["frontend", "ui"],             "max_load": 3},
    {"id": "bob",    "name": "Bob",    "skills": ["backend", "api", "database"],  "max_load": 3},
    {"id": "carol",  "name": "Carol",  "skills": ["devops", "cloud", "security"], "max_load": 3},
    {"id": "david",  "name": "David",  "skills": ["frontend", "backend"],         "max_load": 4},
]

TASK_POOL = {
    "easy": [
        {"id": "t1", "name": "FixLoginButton",  "skills": ["frontend"], "points": 1},
        {"id": "t2", "name": "AddUnitTests",    "skills": ["frontend"], "points": 1},
        {"id": "t3", "name": "UpdateDocs",      "skills": [],           "points": 1},
    ],
    "medium": [
        {"id": "t4",  "name": "BuildAPI",       "skills": ["backend"],           "points": 2},
        {"id": "t5",  "name": "DesignUI",       "skills": ["frontend", "ui"],    "points": 2},
        {"id": "t6",  "name": "DatabaseSchema", "skills": ["backend", "database"],"points": 2},
        {"id": "t7",  "name": "SetupCI",        "skills": ["devops"],            "points": 2},
    ],
    "hard": [
        {"id": "t8",  "name": "FullStackFeature","skills": ["frontend", "backend"],"points": 3},
        {"id": "t9",  "name": "CloudDeploy",    "skills": ["devops", "cloud"],   "points": 3},
        {"id": "t10", "name": "MobileApp",      "skills": ["frontend"],          "points": 3},
        {"id": "t11", "name": "AnalyticsPipeline","skills": ["backend"],         "points": 3},
        {"id": "t12", "name": "SecurityAudit",  "skills": ["security"],          "points": 3},
    ],
}


def _build_tasks(task_index: int) -> List[Dict]:
    if task_index == 0:
        return [dict(t) for t in TASK_POOL["easy"]]
    elif task_index == 1:
        return [dict(t) for t in TASK_POOL["easy"]] + [dict(t) for t in TASK_POOL["medium"]]
    else:
        return (
            [dict(t) for t in TASK_POOL["easy"]]
            + [dict(t) for t in TASK_POOL["medium"]]
            + [dict(t) for t in TASK_POOL["hard"]]
        )


class TaskAllocationEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_index: int = 0
        self._tasks: List[Dict] = []
        self._team: List[Dict] = []
        self._completed: List[str] = []
        self._failed: List[str] = []
        self._total_points: int = 0
        self._earned_points: float = 0.0

    def reset(self, task_index: int = 0, **kwargs: Any) -> TaskAllocationObservation:
        self._task_index = int(task_index) % len(TASK_NAMES)
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._tasks = _build_tasks(self._task_index)
        for t in self._tasks:
            t["status"] = "pending"
            t["assigned_to"] = None

        self._team = [
            {**m, "current_load": 0}
            for m in TEAM
        ]

        self._completed = []
        self._failed = []
        self._total_points = sum(t["points"] for t in self._tasks)
        self._earned_points = 0.0

        return self._make_obs(done=False, reward=0.0, feedback="")

    def step(self, action: TaskAllocationAction, **kwargs: Any) -> TaskAllocationObservation:
        self._state.step_count += 1

        # Parse action content
        try:
            data = json.loads(action.content)
            task_id = str(data.get("task_id", ""))
            agent_id = str(data.get("agent_id", ""))
        except Exception:
            # Try plain "task_id,agent_id" format
            parts = action.content.split(",")
            task_id = parts[0].strip() if parts else ""
            agent_id = parts[1].strip() if len(parts) > 1 else ""

        task = next((t for t in self._tasks if t["id"] == task_id and t["status"] == "pending"), None)
        agent = next((a for a in self._team if a["id"] == agent_id), None)

        reward = 0.0
        feedback = ""

        if task is None:
            reward = -0.05
            feedback = f"Unknown or already-assigned task '{task_id}'."
        elif agent is None:
            reward = -0.05
            feedback = f"Unknown agent '{agent_id}'."
        elif agent["current_load"] >= agent["max_load"]:
            task["status"] = "failed"
            self._failed.append(task_id)
            reward = -0.1
            feedback = f"{agent['name']} is at max capacity. Task '{task_id}' failed."
        else:
            # Check skill match
            required = task["skills"]
            matched = [s for s in required if s in agent["skills"]]
            skill_ratio = len(matched) / len(required) if required else 1.0

            task["status"] = "completed"
            task["assigned_to"] = agent_id
            agent["current_load"] += 1
            self._completed.append(task_id)

            # Reward: skill match quality + task points
            reward = 0.4 * skill_ratio + 0.3
            self._earned_points += task["points"] * skill_ratio
            feedback = (
                f"Assigned '{task['name']}' to {agent['name']}. "
                f"Skill match: {len(matched)}/{len(required) if required else 0}. "
                f"Reward: {reward:.2f}"
            )

        pending = [t for t in self._tasks if t["status"] == "pending"]
        done = len(pending) == 0

        # Final score on episode end
        if done:
            score = min(1.0, self._earned_points / max(self._total_points, 1))
            feedback = (
                f"Episode done! Completed {len(self._completed)}/{len(self._tasks)} tasks. "
                f"Score: {score:.2f}"
            )
            return self._make_obs(done=True, reward=reward, feedback=feedback, score=score)

        return self._make_obs(done=False, reward=reward, feedback=feedback)

    def _make_obs(
        self,
        done: bool,
        reward: float,
        feedback: str,
        score: Optional[float] = None,
    ) -> TaskAllocationObservation:
        pending = [t for t in self._tasks if t["status"] == "pending"]
        available_agents = [a for a in self._team if a["current_load"] < a["max_load"]]

        if score is None:
            score = min(1.0, self._earned_points / max(self._total_points, 1))

        game_state = {
            "pending_tasks": [
                {"id": t["id"], "name": t["name"], "required_skills": t["skills"]}
                for t in pending
            ],
            "team": [
                {
                    "id": a["id"],
                    "name": a["name"],
                    "skills": a["skills"],
                    "load": f"{a['current_load']}/{a['max_load']}",
                }
                for a in available_agents
            ],
            "completed": len(self._completed),
            "failed": len(self._failed),
            "total": len(self._tasks),
        }

        return TaskAllocationObservation(
            task_name=TASK_NAMES[self._task_index],
            game_state=game_state,
            context={
                "task_index": self._task_index,
                "instructions": (
                    'Assign one task to one agent. Respond with JSON: '
                    '{"task_id": "<id>", "agent_id": "<id>"}. '
                    "Match agent skills to task requirements."
                ),
                "difficulty": ["easy", "medium", "hard"][self._task_index],
            },
            step_count=self._state.step_count,
            done=done,
            reward=reward,
            score=score,
            feedback=feedback,
            valid_actions=["allocate"],
        )

    @property
    def state(self) -> State:
        return self._state
