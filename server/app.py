"""
FastAPI application for the Multi-Agent Task Allocation Environment.

Uses openenv-core's create_app factory which provides:
  POST /reset   — reset the environment
  POST /step    — take one action
  GET  /state   — current state
  GET  /schema  — action/observation JSON schemas
  WS   /ws      — WebSocket endpoint for persistent sessions
  GET  /health  — liveness probe
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import TaskAllocationAction, TaskAllocationObservation
    from .task_allocation_environment import TaskAllocationEnvironment
except ImportError:
    from models import TaskAllocationAction, TaskAllocationObservation
    from server.task_allocation_environment import TaskAllocationEnvironment


app = create_app(
    TaskAllocationEnvironment,
    TaskAllocationAction,
    TaskAllocationObservation,
    env_name="multi_agent_task_alloc",
    max_concurrent_envs=4,
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "env": "multi-agent-task-alloc",
        "tasks": ["easy_allocation", "medium_allocation", "hard_allocation"],
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
