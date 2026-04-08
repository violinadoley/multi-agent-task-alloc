"""
inference.py - Multi-Agent Task Allocation inference script.

Runs all 3 tasks (easy_allocation, medium_allocation, hard_allocation) sequentially
using an OpenAI-compatible LLM to generate allocation decisions.

Environment variables
---------------------
API_BASE_URL   OpenAI-compatible API base URL
               (default: https://router.huggingface.co/v1)
MODEL_NAME     Model identifier
               (default: Qwen/Qwen2.5-72B-Instruct)
HF_TOKEN       HuggingFace / API token (required for default endpoint)
IMAGE_NAME     Docker image name (default: multi-agent-task-alloc:latest)
ENV_URL        Override to connect to a running server instead of Docker

Stdout format
-------------
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

from multi_agent_task_alloc import TaskAllocationEnv, TaskAllocationAction

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "no-key"
IMAGE_NAME: str = os.getenv("IMAGE_NAME") or "multi-agent-task-alloc:latest"
ENV_URL: Optional[str] = os.getenv("ENV_URL")

BENCHMARK = "multi-agent-task-alloc"
MAX_STEPS = 15
TEMPERATURE = 0.1
MAX_TOKENS = 256
SUCCESS_THRESHOLD = 0.5

TASKS = [
    {"index": 0, "name": "easy_allocation"},
    {"index": 1, "name": "medium_allocation"},
    {"index": 2, "name": "hard_allocation"},
]

# ─── OpenAI client ────────────────────────────────────────────────────────────

_llm: Optional[OpenAI] = None


def get_llm() -> OpenAI:
    global _llm
    if _llm is None:
        _llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _llm


# ─── Logging ──────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    safe = action.replace("\n", " ").replace("\r", "")[:120]
    err = error if error else "null"
    done_str = "true" if done else "false"
    print(f"[STEP] step={step} action={safe} reward={reward:.2f} done={done_str} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ─── Prompt builder ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a project manager AI. Assign tasks to team members based on their skills.\n"
    "Rules:\n"
    "- Match agent skills to task required_skills for best results.\n"
    "- Don't overload agents (respect their load capacity).\n"
    "- Respond with ONLY valid JSON: {\"task_id\": \"<id>\", \"agent_id\": \"<id>\"}\n"
    "- No explanation, no markdown, just the JSON object."
)


def _build_prompt(obs: Any) -> str:
    state = obs.game_state
    pending = state.get("pending_tasks", [])
    team = state.get("team", [])

    tasks_str = "\n".join(
        f"  - {t['id']}: {t['name']} (requires: {', '.join(t['required_skills']) or 'any'})"
        for t in pending
    )
    team_str = "\n".join(
        f"  - {a['id']}: {a['name']} | skills: {', '.join(a['skills'])} | load: {a['load']}"
        for a in team
    )

    return (
        f"Pending tasks:\n{tasks_str}\n\n"
        f"Available team members:\n{team_str}\n\n"
        f"Pick ONE task and ONE agent. Return JSON only."
    )


def generate_action(obs: Any) -> str:
    prompt = _build_prompt(obs)
    completion = get_llm().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return (completion.choices[0].message.content or "").strip()


# ─── Task runner ─────────────────────────────────────────────────────────────

async def run_task(
    env: TaskAllocationEnv,
    task_index: int,
    task_name: str,
) -> Dict[str, Any]:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    last_error: Optional[str] = None

    try:
        result = await env.reset(task_index=task_index)
        obs = result.observation

        for step_n in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            # Stop if no pending tasks
            if not obs.game_state.get("pending_tasks"):
                break

            try:
                content = generate_action(obs)
            except Exception as exc:
                last_error = f"LLM error: {exc}"
                log_step(step=step_n, action="null", reward=0.0, done=False, error=last_error)
                break

            try:
                result = await env.step(TaskAllocationAction(action_type="allocate", content=content))
            except Exception as exc:
                last_error = f"Env error: {exc}"
                log_step(step=step_n, action=content, reward=0.0, done=False, error=last_error)
                break

            obs = result.observation
            reward = float(result.reward) if result.reward is not None else 0.0
            done = bool(result.done)
            steps_taken = step_n
            rewards.append(reward)
            last_error = None

            log_step(step=step_n, action=content, reward=reward, done=done, error=None)

            if done:
                final_score = float(obs.score) if obs.score is not None else 0.0
                final_score = max(0.0, min(1.0, final_score))
                success = final_score >= SUCCESS_THRESHOLD
                break

    except Exception as exc:
        last_error = str(exc)
        print(f"[DEBUG] run_task error: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return {
        "task_name": task_name,
        "success": success,
        "steps": steps_taken,
        "score": final_score,
        "rewards": rewards,
        "error": last_error,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    if ENV_URL:
        print(f"Connecting to environment at {ENV_URL} ...", flush=True)
        env = TaskAllocationEnv(base_url=ENV_URL)
    else:
        print(f"Starting environment from Docker image: {IMAGE_NAME} ...", flush=True)
        env = await TaskAllocationEnv.from_docker_image(IMAGE_NAME)

    all_results: List[Dict[str, Any]] = []

    try:
        for task in TASKS:
            result = await run_task(env=env, task_index=task["index"], task_name=task["name"])
            all_results.append(result)
            print(flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)

    total = sum(r["score"] for r in all_results) / len(all_results)
    print("=" * 60, flush=True)
    print(f"OVERALL SCORE: {total:.4f}", flush=True)
    for r in all_results:
        tag = "PASS" if r["success"] else "FAIL"
        print(f"  [{tag}] {r['task_name']:25s} score={r['score']:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
