"""OPTIMUS: a NatureBench harness that wraps Claude Code in a keep-alive loop.

Selected with ``--agent optimus``. Built from the run-1 diagnosis of plain
Claude Code + Opus 5.5 on NatureBench-25 (gpu_low):

* Agents stopped as soon as they beat SOTA (g >= 0), often within an hour of a
  4-hour budget, although the leaderboard only counts g > 0.10 as a win.
  -> OPTIMUS states the real objective and resumes the same Claude Code session
     with a status update whenever it ends its turn early, until time runs out.
* Two tasks crashed when the host disk filled. -> a crashed round is resumed,
  the agent sees free disk space in every status update, and is told to keep
  its workspace lean.
* One task was ruled invalid for deriving test targets from test inputs.
  -> explicit validity rules, written from the judge's own policy, are in the
     prompt and repeated on every resume.
* Small per-instance gains were left on the table. -> ``optimus-eval`` snapshots
  each scored output and ``optimus-assemble`` combines the best snapshot per
  instance.

1.1, from the first OPTIMUS run: agents used the full time but spent hours
tuning one stalled method (+0.002 in 2.5 hours, ending at +0.096 on a task where
another method reaches +0.232), and never ended a turn, so guidance sent only on
resume never arrived. Steering now reaches the agent while it works: every
``optimus-eval`` result carries the phase, method-family counts and DIVERSITY /
PLATEAU / THRESHOLD directives, and a PostToolUse hook checks in every ~20
minutes. A clean exit without evaluating is treated as a decline (nudged, then
accepted), not as a crash.

The task prompt is NatureBench's stock Claude Code prompt (so the evaluation
protocol is unchanged) followed by the OPTIMUS operating rules. The evaluation
service, timer, judge and scoring are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .adapter import REGISTRY, AgentAdapter, AgentRunContext
from . import cli_commands as cc
from .claude import ClaudeAgent

OPTIMUS_VERSION = "1.1"
_FILES = (Path(__file__).resolve().parent / "optimus")
_STATE_IN_CONTAINER = "/opt/optimus-state"
_HOME_IN_CONTAINER = "/opt/optimus"

OPTIMUS_RULES = """

# OPTIMUS operating rules
This run is managed by the OPTIMUS harness. These rules take priority over anything above that conflicts with them.

## 1. The objective
Your score is `best_aggregate_improvement` (g). Matching SOTA (g >= 0) is the minimum, not the finish line: on the leaderboard a task only counts as a clear win when **g > +0.10**, and beyond that a larger margin is better. Push g as high as you can for the entire time budget.

## 2. You will use the whole time budget
If you end your turn while time remains, the harness resumes you in the same session with a status update. So never stop to write a final report and never conclude that a score is "good enough"; ending a turn early only wastes time. Run long jobs (training, embedding extraction) in the background with logs, keep the GPU busy, and keep thinking and preparing the next experiment while they run.

## 3. Evaluate with `optimus-eval`, and follow its directives
Use `optimus-eval --family <method-family> "<one line: what changed>"` instead of the raw curl call. It calls the same `/evaluate` endpoint and also records the attempt, its method family and your note in `/workspace/optimus_ledger.md`, snapshots the scored output, and prints your score, best score, per-instance scores, the current phase and any `[OPTIMUS]` directives. `--family` is a short name for the approach (for example `gbm-features`, `gnn`, `esm-embeddings-ridge`, `hmrf-mixture`); variants of the same approach keep the same family. `optimus-status` shows the same information without evaluating. There is no limit on evaluations.
The harness also checks in automatically about every 20 minutes while you work. Lines starting with `[OPTIMUS]` come from the harness: when one contains a directive (DIVERSITY, PLATEAU, THRESHOLD, FINAL), act on it before continuing what you were doing.

## 4. How to search
The run is split into phases by elapsed time: EXPLORE (first ~45%), EXPLOIT (to ~80%), CONSOLIDATE, and FINAL (last 15 minutes).
1. First 15 minutes: read the README and data description, inspect the data, get a simple valid baseline scored.
2. EXPLORE: score at least three genuinely different method families within roughly the first 80 minutes, before polishing any of them. Pick models whose inductive bias matches the data's structure (graphs, spatial neighbourhoods, sequences, images, sets, physical constraints), strong representations (including pretrained models already installed in the environment, used as feature extractors), tree ensembles on well-engineered features, and probabilistic or classical domain algorithms. Run CPU-based families in the background while the GPU trains another.
3. Plateau rule: if your best score has not improved by at least 0.005 in 45 minutes, stop tuning the current approach and try a different family. Small parameter variations of one pipeline are the least valuable use of time: in an earlier run an agent spent its last two and a half hours on variants of one method, gained 0.002 and finished at +0.096, just under the line, on a task where a different method has reached +0.232.
4. EXPLOIT: put effort into the best one or two families with changes that can move the score materially: better representations, longer or better-regularised training, tuning validated on the training data, per-instance specialisation, several seeds.
5. CONSOLIDATE: ensembles and stacking across your strongest families, seed averaging, and per-instance choice of pipeline (check the per-instance scores).
6. FINAL: if the task has two or more instances, run `optimus-assemble --dry-run` and then `optimus-assemble` if it predicts a gain.
Use validation splits of the training data to choose between ideas, and use the evaluation feedback freely to select among fitted models.

## 5. Validity: check before every evaluation
An independent judge reviews your full trace. It judges the attempt that produced your best score, and a later clean attempt cannot replace a tainted best one, so an invalid shortcut costs the whole task. Before every evaluation, make sure that:
- every submitted prediction comes from a method whose key predictive behaviour is trained, learned, fitted or optimised on the provided data;
- test inputs are used only as model inputs at prediction time (plus any unsupervised step the README allows). Never compute labels, targets or target-like statistics from test data: for example, do not infer test outcomes by comparing consecutive test frames or time points, joining or aligning test records, or reading any structure in the test files that encodes the answer. A previous run lost a task exactly this way;
- there are no hard-coded values, lookup tables, hand-set constants or hand-written rules acting as the predictor, and training labels are never copied over as the whole method (nearest neighbours are fine as one learned, validated component);
- no external LLM APIs and no web. Do not try to identify the withheld source paper, and never reproduce its code, checkpoints or reported numbers;
- pretrained models are feature extractors or components, never a ready-made answer for the exact target;
- outputs are never deliberately empty or degenerate.

## 6. Disk and memory
The host disk is shared with other tasks. Keep `/workspace` lean (aim under 40 GB): cache embeddings once in a compact dtype, delete checkpoints and intermediates you no longer need, and check `df -h /workspace` before writing large arrays. Use mixed precision and sensible batch sizes on the 24 GB GPU.

## 7. Keep `run.py` current
At any time, `DATA_DIR=/task/problem/data OUTPUT_DIR=/workspace/output python /workspace/run.py` should regenerate your best pipeline's predictions.
"""


def _task_info(ctx: AgentRunContext) -> dict:
    return {
        "eval_service_url": ctx.eval_service_url,
        "eval_token": ctx.eval_token,
        "time_limit_minutes": ctx.time_limit_minutes,
    }


class OptimusAdapter(AgentAdapter):
    name = "optimus"

    # -- prompt ------------------------------------------------------------
    def system_prompt(self, ctx: AgentRunContext) -> str:
        base = ClaudeAgent(model_name=ctx.model, mode=ctx.mode).build_system_prompt(_task_info(ctx))
        prompt = base + OPTIMUS_RULES
        if ctx.is_resume:
            prompt += (
                "\n\n[RESUME NOTICE] This task was interrupted earlier. /workspace and all prior "
                "evaluations are preserved; read /workspace/optimus_ledger.md and continue."
            )
        return prompt

    # -- host-side state ------------------------------------------------------
    @staticmethod
    def _state_dir(ctx: AgentRunContext) -> Path:
        if ctx.task_out_dir is None:
            raise RuntimeError("OPTIMUS needs task_out_dir")
        return (Path(ctx.task_out_dir) / ".optimus").resolve()

    def build_command(self, ctx: AgentRunContext) -> List[str]:
        state = self._state_dir(ctx)
        state.mkdir(parents=True, exist_ok=True)
        (state / "initial_prompt.md").write_text(ctx.system_prompt, encoding="utf-8")
        claude_argv = cc.build_claude_cmd("__OPTIMUS_PROMPT__", ctx.model)
        config = {
            "optimus_version": OPTIMUS_VERSION,
            "model": ctx.model,
            "timeout_seconds": int(ctx.time_limit_minutes) * 60,
            "claude_argv": claude_argv,
            "is_resume": bool(ctx.is_resume),
            "min_start_seconds": 240,
            "kill_margin_seconds": 45,
            "max_rounds": 60,
        }
        (state / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        claude_state = (Path(ctx.task_out_dir) / ".claude_state").resolve()
        claude_state.mkdir(parents=True, exist_ok=True)
        self._write_hook_settings(claude_state)
        return ["python3", f"{_HOME_IN_CONTAINER}/optimus_loop.py"]

    @staticmethod
    def _write_hook_settings(claude_state: Path) -> None:
        """Register the OPTIMUS PostToolUse check-in hook in the container's ~/.claude/settings.json."""
        path = claude_state / "settings.json"
        settings = {}
        if path.exists():
            try:
                settings = json.loads(path.read_text(encoding="utf-8") or "{}")
            except ValueError:
                settings = {}
        hook = {"type": "command", "command": f"python3 {_HOME_IN_CONTAINER}/optimus_hook.py", "timeout": 15}
        hooks = settings.setdefault("hooks", {})
        entries = [e for e in hooks.get("PostToolUse", []) if "optimus_hook.py" not in json.dumps(e)]
        entries.append({"matcher": "*", "hooks": [hook]})
        hooks["PostToolUse"] = entries
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    def docker_mounts(self, ctx: AgentRunContext) -> List[str]:
        state = self._state_dir(ctx)
        state.mkdir(parents=True, exist_ok=True)
        claude_state = (Path(ctx.task_out_dir) / ".claude_state").resolve()
        claude_state.mkdir(parents=True, exist_ok=True)
        return [
            "-v", f"{_FILES}:{_HOME_IN_CONTAINER}:ro",
            "-v", f"{state}:{_STATE_IN_CONTAINER}",
            # Persist Claude Code sessions on the host (resume + judge transcript).
            "-v", f"{claude_state}:/root/.claude",
        ]

    def extra_env(self, ctx: AgentRunContext) -> List[str]:
        return [
            "-e", f"OPTIMUS_EVAL_TOKEN={ctx.eval_token}",
            "-e", f"OPTIMUS_HOME={_HOME_IN_CONTAINER}",
            "-e", f"OPTIMUS_STATE={_STATE_IN_CONTAINER}",
        ]

    # -- judge transcript ------------------------------------------------------
    def transcript_path(self, task_out_dir: Path) -> Optional[Path]:
        task_out_dir = Path(task_out_dir)
        proj = task_out_dir / ".claude_state" / "projects"
        sids: List[str] = []
        sess_file = task_out_dir / ".optimus" / "sessions.txt"
        if sess_file.exists():
            for line in sess_file.read_text(encoding="utf-8").splitlines():
                sid = line.strip()
                if sid and sid not in sids:
                    sids.append(sid)
        files: List[Path] = []
        if proj.is_dir():
            by_stem = {p.stem: p for p in proj.rglob("*.jsonl")}
            files = [by_stem[s] for s in sids if s in by_stem]
        if len(files) == 1:
            return files[0]
        if len(files) > 1:
            merged = task_out_dir / "optimus_transcript.jsonl"
            with open(merged, "wb") as out:
                for f in files:
                    out.write(f.read_bytes().rstrip(b"\n") + b"\n")
            return merged
        stream = task_out_dir / f"{self.name}.jsonl"
        return stream if stream.exists() else None


REGISTRY.register(OptimusAdapter())
