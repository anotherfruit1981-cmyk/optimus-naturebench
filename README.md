# OPTIMUS: a Claude Code harness for NatureBench

OPTIMUS is an agent adapter for [NatureBench](https://github.com/FrontisAI/NatureBench), the benchmark that sets coding agents against the state of the art reported in Nature-family papers. It runs the Claude Code CLI inside NatureBench's standard task container and keeps the agent working, with a running record of its experiments, for the whole 4-hour budget.

This repository holds the exact code behind our NatureBench Full-track submission: OPTIMUS 1.1 with Claude Opus 5.5 (`anthropic/claude-opus-5.5` through OpenRouter), one frozen configuration for all 90 tasks.

| Full track (90 tasks) | Result |
|---|---|
| Surpass SOTA (valid, g > 0.1) | 30 (33.3%) |
| Match SOTA (valid, g ≥ 0) | 55 (61.1%) |
| Scored | 86 |
| Judged invalid | 4 |
| Not run (DGL wheel server outage) | 3 |

These are our own numbers, computed with NatureBench's `compute_scores.py`; the leaderboard's published figures follow the maintainers' review. The submission's `raw-results/README.md` documents hardware, every infrastructure fault and re-run, the judge, and how OPTIMUS was developed.

## What OPTIMUS adds to stock Claude Code

- **Operating rules** appended to NatureBench's stock task prompt: the objective (g > 0.1 counts as a win), use of the whole budget, a phased search plan, a validity checklist written from the judge's policy, disk hygiene, and keeping `run.py` current.
- **A keep-alive loop.** When Claude Code ends its turn with time left, the loop resumes the same session with a status update, and it stops the agent 45 seconds before the deadline.
- **A PostToolUse hook** that delivers the status and any directives about every 20 minutes.
- **Three in-container tools.** `optimus-eval` calls the same `/evaluate` endpoint as the stock instructions and logs each attempt to a ledger. `optimus-status` prints the ledger. `optimus-assemble` builds a submission from the best-scoring snapshot per instance, which must then be evaluated like any other.

The harness contains no task-specific code, data or settings.

## Layout

```
optimus/                 the harness as run (version 1.1.1; agent code identical to 1.1)
  agent/optimus_adapter.py     NatureBench adapter, prompt and container set-up
  agent/optimus/               loop, hook, ledger and the three tools
  install_optimus.sh           installs the adapter into a NatureBench checkout
  vm_setup.sh                  one-shot set-up for an Ubuntu 22.04 GPU machine
  run_optimus.sh               runs a task list through NatureBench
  README.md                    design notes and operating instructions
operator_scripts/        judging, batching, clean-up and packaging scripts used for the submission
```

See `optimus/README.md` for installation and running.
