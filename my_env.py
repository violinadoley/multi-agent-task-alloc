"""Multi-Agent Task Allocation"""
import random
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from openenv.core import Environment, Action as OpenEnvAction, Observation, State

class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentStatus(str, Enum):
    AVAILABLE = "available"
    BUSY = "busy"

class Skill(BaseModel):
    name: str
    level: int = 3

class TeamMember(BaseModel):
    id: str
    name: str
    skills: List[Skill] = []
    current_load: int = 0
    max_load: int = 3
    status: AgentStatus = AgentStatus.AVAILABLE

class ProjectTask(BaseModel):
    id: str
    name: str
    description: str
    required_skills: List[str] = []
    difficulty: TaskDifficulty
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None

class Action(OpenEnvAction):
    action_type: str = "allocate"
    task_id: str = ""
    agent_id: str = ""
    reasoning: str = ""

class ObservedTask(BaseModel):
    id: str
    name: str
    required_skills: List[str] = []
    status: str

class ObservedMember(BaseModel):
    id: str
    name: str
    skills: List[Dict] = []
    current_load: int
    max_load: int
    status: str

class TaskAllocationObservation(Observation):
    available_tasks: List[ObservedTask] = []
    team_members: List[ObservedMember] = []
    completed_count: int = 0
    failed_count: int = 0
    current_step: int = 0
    max_steps: int = 15
    total_tasks: int = 0
    difficulty: str = ""

class TaskAllocationState(State):
    available_tasks: List[ObservedTask]
    team_members: List[ObservedMember]
    completed_tasks: List[str]
    failed_tasks: List[str]
    current_step: int
    max_steps: int
    total_tasks: int
    episode_info: Dict[str, Any]

class GraderResult(BaseModel):
    score: float
    feedback: str
    tasks_completed: int

TEAM = [
    {"id": "alice", "name": "Alice", "skills": [{"name": "frontend", "level": 5}, {"name": "ui", "level": 5}]},
    {"id": "bob", "name": "Bob", "skills": [{"name": "backend", "level": 5}, {"name": "api", "level": 5}]},
    {"id": "carol", "name": "Carol", "skills": [{"name": "devops", "level": 5}, {"name": "cloud", "level": 5}]},
    {"id": "david", "name": "David", "skills": [{"name": "frontend", "level": 4}, {"name": "backend", "level": 4}]},
]

TASKS = {
    TaskDifficulty.EASY: [
        {"id": "t1", "name": "FixButton", "description": "Fix login button", "required_skills": ["frontend"], "difficulty": TaskDifficulty.EASY},
        {"id": "t2", "name": "AddTests", "description": "Add unit tests", "required_skills": ["frontend"], "difficulty": TaskDifficulty.EASY},
        {"id": "t3", "name": "UpdateDocs", "description": "Update docs", "required_skills": [], "difficulty": TaskDifficulty.EASY},
    ],
    TaskDifficulty.MEDIUM: [
        {"id": "t4", "name": "API", "description": "Create API", "required_skills": ["backend"], "difficulty": TaskDifficulty.MEDIUM},
        {"id": "t5", "name": "UI", "description": "Create UI", "required_skills": ["frontend"], "difficulty": TaskDifficulty.MEDIUM},
        {"id": "t6", "name": "DB", "description": "Database", "required_skills": ["backend"], "difficulty": TaskDifficulty.MEDIUM},
        {"id": "t7", "name": "CI", "description": "CI/CD", "required_skills": ["devops"], "difficulty": TaskDifficulty.MEDIUM},
    ],
    TaskDifficulty.HARD: [
        {"id": "t8", "name": "FullStack", "description": "Full stack", "required_skills": ["frontend", "backend"], "difficulty": TaskDifficulty.HARD},
        {"id": "t9", "name": "Cloud", "description": "Cloud deploy", "required_skills": ["devops", "cloud"], "difficulty": TaskDifficulty.HARD},
        {"id": "t10", "name": "Mobile", "description": "Mobile app", "required_skills": ["frontend"], "difficulty": TaskDifficulty.HARD},
        {"id": "t11", "name": "Analytics", "description": "Analytics", "required_skills": ["backend"], "difficulty": TaskDifficulty.HARD},
        {"id": "t12", "name": "Security", "description": "Security audit", "required_skills": ["backend"], "difficulty": TaskDifficulty.HARD},
    ],
}

class MultiAgentTaskAllocEnv(Environment):
    name = "multi_agent_task_alloc"
    version = "1.0.0"
    MAX_STEPS = 20
    
    def __init__(self, task_difficulty: str = "medium", **kwargs):
        super().__init__(**kwargs)
        self.team = []
        self.tasks = []
        self.completed = []
        self.failed = []
        self.step_num = 0
        self.difficulty = TaskDifficulty(task_difficulty)
        
    async def reset(self) -> TaskAllocationObservation:
        self.team = [TeamMember(**m) for m in TEAM]
        self.tasks = []
        
        if self.difficulty == TaskDifficulty.EASY:
            for t in TASKS[TaskDifficulty.EASY]:
                self.tasks.append(ProjectTask(**t))
        elif self.difficulty == TaskDifficulty.MEDIUM:
            for t in TASKS[TaskDifficulty.EASY]:
                self.tasks.append(ProjectTask(**t))
            for t in TASKS[TaskDifficulty.MEDIUM]:
                self.tasks.append(ProjectTask(**t))
        else:
            for d in TaskDifficulty:
                for t in TASKS[d]:
                    self.tasks.append(ProjectTask(**t))
        
        self.completed = []
        self.failed = []
        self.step_num = 0
        return self._obs()
    
    async def step(self, action: Action) -> tuple[TaskAllocationObservation, float, bool]:
        self.step_num += 1
        reward = 0.0
        
        task = next((t for t in self.tasks if t.id == action.task_id), None)
        agent = next((a for a in self.team if a.id == action.agent_id), None)
        
        if task and agent and task.status == TaskStatus.PENDING:
            has_skill = not task.required_skills or any(s.name in task.required_skills for s in agent.skills)
            
            if has_skill and agent.current_load < agent.max_load:
                task.status = TaskStatus.ASSIGNED
                task.assigned_to = agent.id
                agent.current_load += 1
                
                if task.required_skills:
                    matches = sum(1 for s in agent.skills if s.name in task.required_skills)
                    reward = 0.5 + (matches * 0.2)
                else:
                    reward = 0.5
                
                if random.random() > 0.03:
                    task.status = TaskStatus.COMPLETED
                    self.completed.append(task.id)
                    agent.current_load = max(0, agent.current_load - 1)
                    reward += 0.6
            else:
                task.status = TaskStatus.FAILED
                self.failed.append(task.id)
                reward = -0.1
        else:
            reward = -0.05
        
        pending = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        done = len(pending) == 0 or self.step_num >= self.MAX_STEPS
        
        if self.completed:
            reward += (len(self.completed) / len(self.tasks)) * 0.5
        
        return self._obs(), reward, done
    
    def state(self) -> TaskAllocationState:
        o = self._obs()
        return TaskAllocationState(o.available_tasks, o.team_members, o.completed_tasks, o.failed_tasks, o.current_step, o.max_steps, o.total_tasks, o.episode_info)
    
    def _obs(self) -> TaskAllocationObservation:
        pending = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        return TaskAllocationObservation(
            available_tasks=[ObservedTask(id=t.id, name=t.name, required_skills=t.required_skills, status=t.status.value) for t in pending],
            team_members=[ObservedMember(id=m.id, name=m.name, skills=[{"name": s.name, "level": s.level} for s in m.skills], current_load=m.current_load, max_load=m.max_load, status=m.status.value) for m in self.team],
            completed_count=len(self.completed),
            failed_count=len(self.failed),
            current_step=self.step_num,
            max_steps=self.MAX_STEPS,
            total_tasks=len(self.tasks),
            difficulty=self.difficulty.value
        )
    
    def grade(self, name: str) -> GraderResult:
        c = len(self.completed)
        
        if "easy" in name.lower():
            score = 1.0 if c >= 1 else 0.8
            return GraderResult(score=score, feedback=f"{c}/3", tasks_completed=c)
        elif "medium" in name.lower():
            score = 1.0 if c >= 2 else 0.85
            return GraderResult(score=score, feedback=f"{c}/7", tasks_completed=c)
        else:
            score = 1.0 if c >= 4 else (0.85 if c >= 3 else (0.7 if c >= 2 else 0.5))
            return GraderResult(score=score, feedback=f"{c}/12", tasks_completed=c)

Env = MultiAgentTaskAllocEnv