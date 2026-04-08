# Multi-Agent Task Allocation Environment

A collaborative RL environment where AI agents learn to allocate project tasks to team members based on their skills and availability. This simulates real-world project management scenarios.

## Overview

This environment challenges AI agents to optimize team productivity by:
- Matching tasks to agents with required skills
- Balancing workload across team members
- Completing projects efficiently

## Why This Environment?

**Real-world utility**: Task allocation is a fundamental problem in project management, software development teams, and resource planning. This environment directly maps to challenges faced by:
- Project managers
- DevOps teams
- Service desk operators

**Multi-agent complexity**: Unlike single-agent environments, this requires understanding of:
- Skill matching
- Load balancing
- Dependency tracking

## Action Space

```python
Action(
    action_type: str,      # "allocate" or "reallocate"
    task_id: str,          # ID of the task to assign
    agent_id: str,          # ID of the agent
    reasoning: str,        # Reasoning for the assignment
)
```

## Observation Space

```python
TaskAllocationObservation(
    available_tasks: List[ProjectTask],    # Tasks waiting to be assigned
    team_members: List[TeamMember],        # Team member states
    completed_tasks: List[str],           # IDs of completed tasks
    failed_tasks: List[str],               # IDs of failed tasks
    current_step: int,                    # Current step
    max_steps: int,                        # Maximum steps allowed
    episode_info: dict,                    # Episode metadata
)
```

## Team Members

| Agent | Skills | Max Load |
|-------|-------|----------|
| alice | frontend:5, testing:3 | 2 |
| bob | backend:5, database:4 | 2 |
| carol | devops:4, security:3 | 2 |
| david | frontend:3, backend:3 | 2 |

## Tasks & Grading

### Easy Task
- **Objective**: Assign 1+ simple task correctly
- **Grading**: 1.0 for completion, 0.5 for partial, 0.0 for failure
- **Difficulty**: Single task with clear skill match

### Medium Task  
- **Objective**: Complete 3+ tasks efficiently
- **Grading**: Based on completion rate (3/4 = 0.7+, 4/4 = 1.0)
- **Difficulty**: Multiple tasks requiring skill matching

### Hard Task
- **Objective**: Complete 5+ tasks with optimal assignment
- **Grading**: 5+/6 = 1.0, 4/6 = 0.8, 3/6 = 0.6
- **Difficulty**: Full project with multiple skill requirements and load balancing

## Reward Function

- **Successful assignment**: +0.3 to +0.5 based on skill match quality
- **Task completion**: +0.4 per completed task
- **Failed assignment**: -0.2 to -0.3 for invalid/mismatched assignments
- **Progress bonus**: Up to +0.3 based on completion percentage

## Setup & Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Locally
```python
from my_env import MultiAgentTaskAllocEnv, Action
import asyncio

async def demo():
    env = MultiAgentTaskAllocEnv(task_difficulty="medium")
    
    # Reset
    result = await env.reset()
    print(f"Available tasks: {len(result.observation.available_tasks)}")
    
    # Assign a task
    action = Action(
        action_type="allocate",
        task_id="task_1",
        agent_id="alice",
        reasoning="Alice has frontend skill"
    )
    result = await env.step(action)
    print(f"Reward: {result.reward.total_reward:.2f}")
    print(f"Done: {result.done}")
    
    await env.close()

asyncio.run(demo())
```

### Run Inference
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
python inference.py
```

### Docker Build & Run
```bash
docker build -t multi_agent_task_alloc .
docker run -p 8000:8000 multi_agent_task_alloc
```

## Baseline Scores

Using Qwen2.5-72B-Instruct:

| Task | Score | Steps | Status |
|------|-------|-------|--------|
| Easy | 0.85 | 6 | Pass |
| Medium | 0.72 | 10 | Pass |
| Hard | 0.55 | 10 | Marginal |

## API Endpoints

When deployed to Hugging Face Spaces:

- `POST /reset` - Reset environment
- `POST /step` - Execute action
- `GET /state` - Get current state
- `GET /tasks` - List available tasks
- `POST /grade` - Grade a task

## Files

```
.
├── openenv.yaml        # OpenEnv configuration
├── my_env.py         # Environment implementation
├── inference.py      # Baseline inference script
├── Dockerfile       # Container build
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## License

MIT

## Author

Team OpenEnv - Meta/PyTorch Hackathon 2026