"""
OpenEnv Server for Multi-Agent Task Allocation
=====================================
Serves the environment via HTTP API for Hugging Face Spaces deployment.
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from my_env import MultiAgentTaskAllocEnv, Action


app = FastAPI(title="Multi-Agent Task Allocation Environment")
env: Optional[MultiAgentTaskAllocEnv] = None


class ResetResponse(BaseModel):
    observation: dict


class StepRequest(BaseModel):
    action: dict


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool


class StateResponse(BaseModel):
    state: dict


class GradeRequest(BaseModel):
    task_name: str


class GradeResponse(BaseModel):
    score: float
    feedback: str
    tasks_completed: int


@app.on_event("startup")
async def startup():
    global env
    env = MultiAgentTaskAllocEnv()


@app.post("/reset", response_model=ResetResponse)
async def reset():
    global env
    if env is None:
        env = MultiAgentTaskAllocEnv()
    obs = await env.reset()
    return ResetResponse(observation=obs.model_dump())


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    
    try:
        action = Action(**request.action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
    
    obs, reward, done = await env.step(action)
    
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done
    )


@app.get("/state", response_model=StateResponse)
async def state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    
    st = env.state()
    return StateResponse(state=st.model_dump())


@app.get("/tasks")
async def tasks():
    return {"tasks": [
        {"name": "task_easy", "difficulty": "easy"},
        {"name": "task_medium", "difficulty": "medium"},
        {"name": "task_hard", "difficulty": "hard"},
    ]}


@app.post("/grade", response_model=GradeResponse)
async def grade(request: GradeRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    
    result = env.grade(request.task_name)
    return GradeResponse(
        score=result.score,
        feedback=result.feedback,
        tasks_completed=result.tasks_completed
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "environment": "multi_agent_task_alloc"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
