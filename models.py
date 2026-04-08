from __future__ import annotations

from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TaskAllocationAction(Action):
    action_type: str = Field(default="allocate", description="allocate | plan | optimize")
    content: str = Field(..., description='JSON string: {"task_id": "t1", "agent_id": "alice"}')


class TaskAllocationObservation(Observation):
    task_name: str = Field(default="")
    game_state: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    step_count: int = Field(default=0)
    score: float = Field(default=0.0)
    feedback: str = Field(default="")
    valid_actions: List[str] = Field(default_factory=list)
