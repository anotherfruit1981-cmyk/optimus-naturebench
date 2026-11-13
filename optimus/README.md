# OPTIMUS 1.1: a NatureBench harness for Claude Code

OPTIMUS wraps Claude Code in a keep-alive loop and gives it explicit rules. Select it with `--agent optimus`. The task prompt, evaluation service, timer, validity judge and scoring are NatureBench's own and are unchanged. Built against NatureBench `main` at commit f92b849 (20 Oct 2026).

## What changed in 1.1, and why

The first OPTIMUS run (1.0, stopped after 8 of 22 tasks) showed that keeping agents working for the full 4 hours was not enough.

- **Agents stalled on one method.** On tissue neighbourhood detection the agent worked for 223 minutes over 138 attempts, but spent its last two and a half hours tuning one pipeline, gaining 0.002 and finishing at +0.096, just under the surpass line, on a task where HELIX reaches +0.232. On spatial multi-omics it finished at +0.098, while run 1, using a different method, had reached +0.215 in 16 minutes.
- **Guidance never arrived.** Neither agent ended its turn, so 1.0's phase guidance, which was only sent on resume, never reached it.
- **Refusals were treated as crashes.** On spatial domain identification, which reached the ceiling (ARI 0.99) in 10 minutes, the agent kept declining to continue. 1.0 treated each refusal as a crash, backing off and opening fresh sessions 21 times.

1.1 changes three things.

1. **Steering while the agent works.** Every `optimus-eval` result now prints the phase (EXPLORE, then EXPLOIT, then CONSOLIDATE, then FINAL), the method families scored and their best scores, and directives:
   - **DIVERSITY:** fewer than three families scored by about 36 minutes in, or overdue after 80;
   - **PLATEAU:** the best score has gained less than 0.005 in 45 minutes, so the next two experiments must be new families, or ensembles if the run is in CONSOLIDATE;
   - **THRESHOLD:** the score is between 0 and +0.10;
   - **FINAL:** the last 15 minutes.

   A Claude Code PostToolUse hook delivers the same check-in about every 20 minutes when there have been no evaluations, so agents that rarely evaluate get steered too.
2. **Method families.** `optimus-eval --family <name> "<note>"` tags each attempt, and the ledger and directives track families.
3. **Declines are not crashes.** A clean exit without a new evaluation gets one nudge, which asks the agent to state `CEILING:` and a reason if the task cannot improve. A third consecutive decline, or a ceiling claim at g ≥ 0, ends the loop. Only non-zero exits are retried as failures.

Tested with the real Claude Code CLI (2.1.316) against a mock Anthropic API: the hook fires in `-p` mode and its check-in reaches the model's next request; `optimus-eval` steering appears in tool results; `--session-id` and `--resume` keep one session file. The decline and ceiling paths and the steering scenarios were tested separately.

## Why it exists: what run 1 showed

Run 1 used NatureBench's stock Claude Code harness with Opus 5.5 on the 22 gpu_low tasks of NatureBench-25.

| | Surpassed (g > 0.1) | Matched (g ≥ 0) |
|---|---|---|
| Run 1 (Claude Code + Opus 5.5) | 4 | 13 |
| HELIX + Opus 5 (leaderboard #1, same 22 tasks) | 6 | 16 |
| EvoRA + gpt-6.1-sol (NatureBench-25 #1, same 22 tasks) | 7 | 17 |

The run-1 losses had four causes.

1. **Stopping at "beat SOTA".** 7 of 20 completed tasks ended within an hour of a 4-hour budget, and 10 within two hours. A typical final message read: "My method now beats the paper's best reported result... finishing in well under the 240 minutes." The leaderboard only counts g > 0.10. Metabolomic profile prediction stopped at 35 minutes on +0.073; HELIX reached +0.339 on it.
2. **Unused time even on long runs.** The long-running agents stopped around 210 minutes, and several were still improving when they did. Molecular property prediction reached its best, +0.099, in its final minute, just short of the 0.1 line.
3. **Infrastructure.** Two tasks crashed when the 200 GB disk filled. One of them had written 13 GB of protein embeddings.
4. **Validity.** One task was ruled invalid because it reconstructed test targets by comparing consecutive frames of the test embryo.

Run-1 API cost: about $100 for 22 tasks, $4.60 per task on average.

## What OPTIMUS changes

| Run-1 problem | OPTIMUS response |
|---|---|
| Agents stop once g ≥ 0 | The prompt states that only g > 0.10 counts and that the full budget will be used. Whenever the agent ends its turn early, the loop resumes the **same session** with a status update (time left, best g and gap to 0.10, weakest instances, ledger) and phase guidance: explore, then exploit, then consolidate, then a final step. |
| Time left unused | Rounds continue until about 4 minutes remain. A watchdog stops Claude Code 45 seconds before the deadline. |
| Crashes | A crashed round is resumed. After two fast failures the loop opens a fresh session with a restart notice. Free disk space appears in every status update, with a warning below 20 GB. |
| Invalid shortcut | Validity rules taken from the judge's own policy, including the run-1 failure case, are in the prompt and repeated on every resume. The agent is told to check before each evaluation, because the judged attempt is the best one and cannot be replaced later. |
| Per-instance gains left unused | `optimus-eval` snapshots every scored output. `optimus-assemble` builds a submission from the best snapshot for each instance and writes a manifest. On run-1 data this alone would have lifted molecular property prediction from 0.099 to 0.102. |

The tools available inside the container are `optimus-eval --family <name> "<note>"`, `optimus-status` and `optimus-assemble [--dry-run]`. The agent's experiment log is `/workspace/optimus_ledger.md`.

The judge reads one continuous transcript: OPTIMUS records every session it opens and merges them in order for the judge.

## Files

- `agent/optimus_adapter.py`: the NatureBench adapter (prompt, mounts, environment, judge transcript).
- `agent/optimus/optimus_loop.py`: the keep-alive loop that runs in the container.
- `agent/optimus/optimus_common.py`, `agent/optimus/bin/*`: the in-container tools and steering logic.
- `agent/optimus/optimus_hook.py`: the Claude Code PostToolUse check-in hook.
- `install_optimus.sh`: installs OPTIMUS into a NatureBench checkout and applies the OpenRouter judge fix.
- `vm_setup.sh`: one-shot setup of a fresh Vast.ai Ubuntu 22.04 VM, including OPTIMUS.
- `run_optimus.sh`: launches a run with your OpenRouter key.
- `status.py`: progress check: `python3 ~/status.py [results-name]`.

## Tested

The loop and tools were tested against a mock evaluation service and a mock Claude Code CLI. Those tests covered early stops, a crashed round, a lost session with fresh-session recovery, the final-phase exit, snapshot pruning, per-instance assembly, and transcript merging through the judge's own parser. They have **not** yet run inside the real task container, so the first real run is also the integration test. Check the first status update about 30 minutes in.

## Running it

1. Rent a Vast.ai "Ubuntu 22.04 VM" with 4× RTX 4090, **at least 500 GB disk**, and reliability of 99% or higher. Add your SSH key before renting.
2. Copy the bundle to the VM and run the setup:
   ```
   scp -P <port> ~/Downloads/optimus_bundle.tar.gz root@<ip>:~/
   ssh -o ServerAliveInterval=30 -p <port> root@<ip>
   tar xzf optimus_bundle.tar.gz && bash ~/optimus/vm_setup.sh
   ```
3. Save the OpenRouter key, open tmux and start the 22 gpu_low tasks:
   ```
   tmux new -s optimus
   ~/run_optimus.sh task-set/naturebench-25/gpu_low.txt optimus_nb25_gpu_low 0,1,2,3
   ```
4. Optionally, in a second tmux window (Ctrl+B then C), run the one CPU task without a GPU, as the reference setup does:
   ```
   ~/run_optimus.sh task-set/naturebench-25/cpu.txt optimus_nb25_cpu
   ```
5. Check progress with `python3 ~/status.py optimus_nb25_gpu_low`.

The two gpu_high tasks need an A100 80 GB VM, about 4.5 hours with two GPUs or 8.5 hours with one. Run them with `~/run_optimus.sh task-set/naturebench-25/gpu_high.txt optimus_nb25_gpu_high 0`.

## Budget

Expect roughly $10 of API spend per task now that agents use the full 4 hours, so about $230 to $300 for the 23 tasks above. Make sure your OpenRouter workspace budget has at least $350 remaining before starting: a run that hits the cap mid-way loses the unfinished tasks.
