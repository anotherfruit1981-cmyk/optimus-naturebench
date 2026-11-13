#!/usr/bin/env python3
"""Claude Code PostToolUse hook: periodic OPTIMUS steering while the agent works.

Registered in the task container's ~/.claude/settings.json by the OPTIMUS
adapter. Claude Code runs it after every tool call; it stays silent unless no
steering has been shown for OPTIMUS_HOOK_EVERY_S seconds (default 20 minutes),
or the run has just entered its final phase. When it speaks, its message is
added to the agent's context as hook output.

Must be fast and must never fail the tool call: every error path exits 0
silently.
"""
import json
import os
import sys
import time


def main() -> int:
    try:
        sys.stdin.read()  # event payload; not needed beyond draining stdin
    except Exception:  # noqa: BLE001
        pass
    try:
        os.environ.setdefault("OPTIMUS_HTTP_TIMEOUT", "4")
        sys.path.insert(0, os.environ.get("OPTIMUS_HOME", "/opt/optimus"))
        import optimus_common as oc  # noqa: E402

        every = float(os.environ.get("OPTIMUS_HOOK_EVERY_S", 1200))
        state_path = oc.STEER_STATE
        state = {}
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8") or "{}")
        last = float(state.get("last_steer", 0))
        tr = oc.time_remaining() or {}
        rem = tr.get("remaining_seconds")
        entering_final = (rem is not None and float(rem) <= oc.FINAL_MIN * 60
                          and not state.get("final_announced"))
        if time.time() - last < every and not entering_final:
            return 0
        lines = oc.steering_lines()
        if not lines:
            return 0
        if entering_final:
            state["final_announced"] = True
        state["last_steer"] = time.time()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state), encoding="utf-8")
        msg = ("OPTIMUS harness check-in (automatic, while you work):\n" + "\n".join(lines)
               + "\nFollow any [OPTIMUS] directive above before continuing your current line of work.")
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}}))
    except Exception:  # noqa: BLE001
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
