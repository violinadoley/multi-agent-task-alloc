---
title: Multi-Agent Task Allocation
emoji: 👥
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "1.0"
app_file: openenv_server.py
pinned: false
---

# Multi-Agent Task Allocation Environment

A collaborative RL environment where AI agents allocate project tasks to team members based on their skills.

## Overview

This environment challenges AI agents to optimize team productivity by:
- Matching tasks to agents with required skills
- Balancing workload across team members
- Completing projects efficiently

## Team Members

| Agent | Skills |
|-------|--------|
| alice | frontend, ui |
| bob | backend, api |
| carol | devops, cloud |
| david | frontend, backend |

## Tasks & Grading

### Easy Task
- **Objective**: Complete 1+ simple task
- **Grading**: Score 1.0 if completed ≥1

### Medium Task
- **Objective**: Complete 2+ tasks
- **Grading**: Score 1.0 if completed ≥2

### Hard Task
- **Objective**: Complete 4+ tasks
- **Grading**: Score 1.0 if completed ≥4

## API Endpoints

- `POST /reset` - Reset environment
- `POST /step` - Execute action
- `GET /state` - Get current state
- `POST /grade` - Grade a task

## Setup

```bash
docker build -t multi_agent_task_alloc .
docker run -p 8000:8000 multi_agent_task_alloc
```
