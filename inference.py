"""
Inference Script for Multi-Agent Task Allocation Environment
=========================================================
Uses OpenAI Client to run a model against the environment.
"""

import asyncio
import os
import json
import random
from typing import List, Optional

from openai import OpenAI

from my_env import (
    MultiAgentTaskAllocEnv,
    Action,
    TaskDifficulty,
)

# Configuration
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASKS = [
    {"name": "task_easy", "difficulty": TaskDifficulty.EASY, "display": "Easy Task Allocation"},
    {"name": "task_medium", "difficulty": TaskDifficulty.MEDIUM, "display": "Medium Task Allocation"},
    {"name": "task_hard", "difficulty": TaskDifficulty.HARD, "display": "Hard Task Allocation"},
]

MAX_STEPS = 10
BENCHMARK = "multi_agent_task_alloc"

SYSTEM_PROMPT = """You are a project manager AI. Assign tasks to team members who have the required skills.

Team members:
- alice: frontend, testing
- bob: backend, database
- carol: devops, security
- david: frontend, backend

Reply with JSON: {"task_id": "X", "agent_id": "Y", "reasoning": "Z"}
Only respond with valid JSON.
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_model_action(client: OpenAI, observation_json: str) -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Tasks and team:\n{observation_json}"},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        response = (completion.choices[0].message.content or "").strip()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[^}]+\}', response)
            if match:
                return json.loads(match.group())
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", flush=True)
    return {}


async def run_task(client: OpenAI, task_config: dict) -> dict:
    task_name = task_config["name"]
    difficulty = task_config["difficulty"]
    
    env = MultiAgentTaskAllocEnv(task_difficulty=difficulty.value)
    history: List[float] = []
    
    obs = await env.reset()
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    for step in range(1, MAX_STEPS + 1):
        obs_dict = {
            "available_tasks": [{"id": t.id, "name": t.name, "skills": t.required_skills} for t in obs.available_tasks],
            "team_members": [{"id": m.id, "skills": [s["name"] for s in m.skills], "load": m.current_load} for m in obs.team_members],
        }
        
        action_dict = get_model_action(client, json.dumps(obs_dict))
        
        task_id = action_dict.get("task_id")
        agent_id = action_dict.get("agent_id")
        reasoning = action_dict.get("reasoning", "auto")
        
        if not task_id and obs.available_tasks:
            task_id = random.choice(obs.available_tasks).id
        if not agent_id:
            agent_id = random.choice(obs.team_members).id
        
        action = Action(
            action_type="allocate",
            task_id=task_id or "task_1",
            agent_id=agent_id or "alice",
            reasoning=reasoning,
        )
        
        obs, reward, done = await env.step(action)
        history.append(reward)
        
        action_str = f'allocate("{action.task_id}", "{action.agent_id}")'
        error = None if reward >= 0 else "invalid"
        log_step(step=step, action=action_str, reward=reward, done=done, error=error)
        
        if done:
            break
    
    grader_result = env.grade(task_name)
    score = grader_result.score
    
    await env.close()
    
    success = score >= 0.5
    log_end(success=success, steps=len(history), score=score, rewards=history)
    
    return {"task": task_name, "score": score, "steps": len(history), "success": success}


async def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN not set")
        return
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results = []
    
    for task_config in TASKS:
        result = await run_task(client, task_config)
        results.append(result)
        await asyncio.sleep(1)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for r in results:
        print(f"{r['task']}: score={r['score']:.3f}, steps={r['steps']}, success={r['success']}")
    
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\nAverage Score: {avg_score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
