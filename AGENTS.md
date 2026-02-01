# Agent Configuration

> This file defines the sub-agents available in the harness.

## Available Agents

| Agent | Role | Trigger |
|-------|------|---------|
| Planner | Converts ideas into structured plans | `/plan-with-agent` |
| Worker | Executes planned tasks | `/work` |
| Reviewer | 8-expert code review | `/harness-review` |
| Fixer | Automatic error correction | On test/lint failure |

## Agent Memory

Agents share context through:
- `Plans.md` - Current plans and progress
- `CLAUDE.md` - Project-specific instructions
- `.claude/` - Session memory and settings

---

## Custom Agent Profiles

_Add custom agent profiles in `profiles/` directory_
