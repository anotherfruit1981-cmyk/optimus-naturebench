#!/usr/bin/env python3
"""OPTIMUS keep-alive loop: runs Claude Code until the task's time budget is spent.

Launched by NatureBench (``docker exec ... python3 /opt/optimus/optimus_loop.py``)
in place of a single ``claude -p`` call. Each round runs Claude Code in the same
session; when it ends its turn with time left, the loop resumes the session with
a status update and phase guidance. Claude's stream-json goes to stdout (the
NatureBench agent log); the loop's own messages go to stderr and loop.log.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

OPTIMUS_HOME = os.environ.get("OPTIMUS_HOME", "/opt/optimus")
STATE = Path(os.environ.get("OPTIMUS_STATE", "/opt/optimus-state"))
sys.path.insert(0, OPTIMUS_HOME)
import optimus_common as oc  # noqa: E402

CFG = json.loads((STATE / "config.json").read_text(encoding="utf-8"))
TIMEOUT_S = float(CFG.get("timeout_seconds", 14400))
MIN_START_S = float(os.environ.get("OPTIMUS_MIN_START_S", CFG.get("min_start_seconds", 240)))
KILL_MARGIN_S = float(os.environ.get("OPTIMUS_KILL_MARGIN_S", CFG.get("kill_margin_seconds", 45)))
MAX_ROUNDS = int(os.environ.get("OPTIMUS_MAX_ROUNDS", CFG.get("max_rounds", 60)))
FAST_FAIL_S = float(os.environ.get("OPTIMUS_FAST_FAIL_S", 90))
FINAL_S = float(os.environ.get("OPTIMUS_FINAL_S", 900))
BACKOFF_S = float(os.environ.get("OPTIMUS_BACKOFF_S", 30))
POLL_S = float(os.environ.get("OPTIMUS_POLL_S", 30))
LOG = STATE / "loop.log"
SESSIONS = STATE / "sessions.txt"

T0 = time.time()


def log(msg: str) -> None:
    line = f"[optimus {time.strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def remaining_s() -> float:
    tr = oc.time_remaining()
    if tr and tr.get("remaining_seconds") is not None:
        try:
            return float(tr["remaining_seconds"])
        except (TypeError, ValueError):
            pass
    return TIMEOUT_S - (time.time() - T0)


def total_attempts() -> int:
    bs = oc.best_score() or {}
    try:
        return int(bs.get("total_attempts") or 0)
    except (TypeError, ValueError):
        return 0


def claude_argv(prompt: str, sid: str, resume: bool) -> List[str]:
    argv = [a if a != "__OPTIMUS_PROMPT__" else prompt for a in CFG["claude_argv"]]
    # argv[0] is "claude"; session flags go right after it.
    head, rest = argv[:1], argv[1:]
    return head + (["--resume", sid] if resume else ["--session-id", sid]) + rest


PHASE_GUIDE = {
    "EXPLORE": ("PHASE EXPLORE: score genuinely different method families before polishing any of them "
                "(a model whose inductive bias matches the data's structure, strong pretrained or engineered "
                "representations, tree ensembles, probabilistic or classical domain methods). Tag each with "
                "--family."),
    "EXPLOIT": ("PHASE EXPLOIT: improve your best one or two families with changes that can move the score by "
                "more than a few thousandths: better representations, longer or better-regularised training, "
                "per-instance specialisation, several seeds."),
    "CONSOLIDATE": ("PHASE CONSOLIDATE: combine your strongest valid models across families (ensembles, "
                    "stacking, seed averaging, per-instance choice of pipeline). Avoid large rewrites."),
    "FINAL": ("PHASE FINAL: finish only what can be evaluated in time, run optimus-assemble if it predicts a "
              "gain, keep run.py reproducing your best pipeline, then end your turn."),
}


def continuation_prompt(round_idx: int, restart: bool, declined: int = 0) -> str:
    rem = remaining_s()
    st = oc.steering()
    parts = []
    if restart:
        parts.append("[OPTIMUS RESTART] The previous Claude Code session ended with an infrastructure "
                     "error. Your /workspace files and all evaluation history are intact. Read "
                     "/workspace/optimus_ledger.md and continue from your best approach.")
    parts.append(f"[OPTIMUS] Round {round_idx}. You ended your turn but time remains, so you are being "
                 "resumed. Status:")
    parts.extend("  " + s for s in oc.status_lines(max_ledger=8))
    parts.extend(oc.steering_lines(st))
    if st.get("phase") in PHASE_GUIDE:
        parts.append(PHASE_GUIDE[st["phase"]])
    if declined:
        parts.append("[OPTIMUS] Your last turn ended without a new evaluation. If you genuinely believe this "
                     "task is at its ceiling (the metric cannot meaningfully improve), reply with one line "
                     "starting 'CEILING:' and the reason, and end your turn. Otherwise run the next experiment "
                     "now.")
    free = oc.disk_free()
    if 0 <= free < 20e9:
        parts.append(f"DISK WARNING: only {oc.gb(free)} free. Delete large intermediates you no longer "
                     "need (old checkpoints, cached arrays) before generating more.")
    reminder = ("Reminders: every prediction must come from a method fitted on the provided training "
                "data; never derive targets or target-like statistics from test inputs; evaluate with "
                "`optimus-eval --family <family> \"<what changed>\"`.")
    if rem > FINAL_S:
        reminder += (" Do not write a final report: keep working until the time is nearly gone. "
                     "Ending your turn early only costs time.")
    parts.append(reminder)
    oc.mark_steered()
    return "\n".join(parts)


def run_round(argv: List[str], env: dict) -> tuple:
    """Run one Claude Code round, streaming stdout, killing it near the deadline."""
    t_start = time.time()
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", env=env, start_new_session=True)
    seen_sid: List[str] = []
    ceiling: List[str] = []
    stop = threading.Event()

    def watchdog() -> None:
        while not stop.wait(POLL_S):
            if remaining_s() < KILL_MARGIN_S:
                log("deadline reached: stopping Claude Code")
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    pass
                if not stop.wait(20):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        pass
                return

    err_text: List[str] = []

    def pump_err() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            sys.stderr.write(line)
            if len(err_text) < 20 and line.strip():
                err_text.append(line.strip()[:300])
        sys.stderr.flush()

    threading.Thread(target=watchdog, daemon=True).start()
    t_err = threading.Thread(target=pump_err, daemon=True)
    t_err.start()
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if '"session_id"' in line and not seen_sid:
            try:
                obj = json.loads(line)
                if obj.get("session_id"):
                    seen_sid.append(obj["session_id"])
            except ValueError:
                pass
        if "CEILING:" in line and '"type":"result"' in line.replace(" ", "") and not ceiling:
            ceiling.append("ceiling")
        if "No conversation found" in line or '"is_error":true' in line.replace(" ", ""):
            if len(err_text) < 20:
                err_text.append(line.strip()[:300])
    rc = proc.wait()
    stop.set()
    t_err.join(timeout=5)
    return rc, time.time() - t_start, (seen_sid[0] if seen_sid else None), err_text, bool(ceiling)


def main() -> int:
    initial = (STATE / "initial_prompt.md").read_text(encoding="utf-8")
    env = dict(os.environ)
    env["PATH"] = f"{OPTIMUS_HOME}/bin:" + env.get("PATH", "")
    env.setdefault("BASH_MAX_TIMEOUT_MS", "3600000")
    log(f"OPTIMUS start: model={CFG.get('model')} timeout={TIMEOUT_S:.0f}s remaining={remaining_s():.0f}s")

    sid: Optional[str] = None
    fresh = True
    restart = bool(CFG.get("is_resume"))
    fails = 0
    declined = 0
    round_idx = 0
    while round_idx < MAX_ROUNDS:
        rem = remaining_s()
        if rem < MIN_START_S:
            log(f"{rem:.0f}s left (< {MIN_START_S:.0f}s): not starting another round")
            break
        final_phase = (not fresh) and rem <= FINAL_S
        if fresh:
            sid = str(uuid.uuid4())
            prompt = initial if not restart else initial + "\n\n" + continuation_prompt(round_idx, True)
        else:
            prompt = continuation_prompt(round_idx, False, declined)
        with open(SESSIONS, "a", encoding="utf-8") as fh:
            fh.write(f"{sid}\n")
        attempts_before = total_attempts()
        log(f"round {round_idx}: {'new session' if fresh else 'resume'} {sid}, {rem / 60:.1f} min left")
        rc, dur, stream_sid, errs, ceiling = run_round(claude_argv(prompt, sid, resume=not fresh), env)
        attempts_after = total_attempts()
        log(f"round {round_idx} ended: rc={rc} {dur / 60:.1f} min, attempts {attempts_before}->{attempts_after}")
        if stream_sid and stream_sid != sid:
            log(f"session id changed {sid} -> {stream_sid}")
            sid = stream_sid
        round_idx += 1
        if final_phase and rc == 0:
            log("final-phase round completed normally: done")
            break

        if ceiling and declined >= 1:
            best_now = (oc.best_score() or {}).get("best_aggregate_improvement")
            if best_now is not None and best_now >= 0:
                log(f"agent declared CEILING at g={best_now:+.4f}: done")
                break
        # A clean exit (rc 0) with no new evaluation is a decline, not a crash: nudge once, and if the agent
        # declines again, accept it. Only non-zero exits are treated as infrastructure failures.
        if rc == 0 and attempts_after == attempts_before and dur < 600 and not fresh:
            declined += 1
            log(f"agent ended its turn without evaluating (decline #{declined})")
            if declined >= 3:
                log("agent declined repeatedly: done")
                break
            fails = 0
            continue
        if attempts_after > attempts_before:
            declined = 0
        failed = rc != 0 and dur < FAST_FAIL_S
        if fresh and dur < FAST_FAIL_S and attempts_after == attempts_before:
            failed = True
        if failed:
            fails += 1
            log(f"fast failure #{fails} (rc={rc}) {' | '.join(errs)[:400]}")
            if fails >= 2 or any("No conversation found" in e for e in errs):
                # Session may be unusable: start a fresh one with the restart notice.
                fresh, restart = True, True
            wait = min(BACKOFF_S * (2 ** (fails - 1)), 10 * BACKOFF_S, max(0.0, remaining_s() - MIN_START_S))
            if wait > 0:
                log(f"backing off {wait:.0f}s")
                time.sleep(wait)
            continue
        fails = 0
        fresh, restart = False, False
    log("OPTIMUS finished")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
