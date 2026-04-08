from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TaskAllocationAction, TaskAllocationObservation


class TaskAllocationEnv(EnvClient[TaskAllocationAction, TaskAllocationObservation, State]):

    def _step_payload(self, action: TaskAllocationAction) -> Dict:
        return {
            "action_type": action.action_type,
            "content": action.content,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TaskAllocationObservation]:
        obs_data = payload.get("observation", {})
        observation = TaskAllocationObservation(
            task_name=obs_data.get("task_name", ""),
            game_state=obs_data.get("game_state", {}),
            context=obs_data.get("context", {}),
            step_count=obs_data.get("step_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            score=obs_data.get("score", 0.0),
            feedback=obs_data.get("feedback", ""),
            valid_actions=obs_data.get("valid_actions", []),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
