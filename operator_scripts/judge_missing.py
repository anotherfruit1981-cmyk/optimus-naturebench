#!/usr/bin/env python3
"""Judge finished tasks that never got a verdict (their run was stopped before its judge stage).

    python ~/judge_missing.py results/v11_full_gpu_low <task> [<task> ...]

Same judge as solve.py's stage: NatureBench's judge_task with openai/gpt-5.5, same evidence, prompt,
policy and parser, called through OpenRouter. One retry on an API error. If OpenAI's bio_policy filter
refuses, it falls back to anthropic/claude-sonnet-5.5 (the rule used for s41592-023-02032-5) and records
why in judge_note. Tasks that already have a verdict are skipped.
"""
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NB = Path.home() / "NatureBench"
OR = "https://openrouter.ai/api"
STOCK = "openai/gpt-5.5"
FALLBACK = "anthropic/claude-sonnet-5.5"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    out = NB / sys.argv[1]
    tasks = sys.argv[2:]
    key = (Path.home() / ".openrouter_key").read_text().strip()
    os.environ.update(JUDGE_MODEL=STOCK, JUDGE_OPENAI_BASE_URL=OR, JUDGE_OPENAI_API_KEY=key,
                      JUDGE_ANTHROPIC_BASE_URL=OR, JUDGE_ANTHROPIC_API_KEY=key)
    os.chdir(NB)
    sys.path.insert(0, str(NB))
    import judge_core.clients as cl
    from judge import judge_task

    local = threading.local()

    def call(user_prompt, model=None, base_url=None, api_key=None, timeout=cl.DEFAULT_TIMEOUT):
        body = {"model": model, "instructions": cl.JUDGE_SYSTEM, "input": user_prompt,
                "max_output_tokens": cl.MAX_JUDGE_OUTPUT_TOKENS}
        req = urllib.request.Request(cl.build_api_url(base_url, "/v1/responses"), data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
        local.err = None
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            local.err = f"HTTP {e.code}"
            return False, f"HTTP {e.code}: request failed"
        except Exception as e:  # noqa: BLE001
            local.err = type(e).__name__
            return False, f"{type(e).__name__}: {e}"
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict):
            local.err = err.get("code") or "error"
        out_items = payload.get("output", []) if isinstance(payload, dict) else []
        text = "".join(str(c.get("text", "")) for i in out_items if isinstance(i, dict)
                       for c in (i.get("content") or []) if isinstance(c, dict) and c.get("type") == "output_text")
        return True, text

    cl.call_openai_judge = call
    cl.call_anthropic_judge = call

    def one(t: str) -> str:
        d = out / t
        jv = d / "judge_verdict.json"
        if jv.exists():
            return f"{t}: already judged, skipped"
        if not (d / "submissions.jsonl").is_file():
            return f"{t}: no submissions, skipped"
        problem = NB / "data" / "naturebench_data" / "tasks" / t / "problem"
        v = judge_task(t, d, "optimus", model=STOCK, task_problem_dir=problem)
        if v.get("is_valid") is None and getattr(local, "err", None) not in (None, "bio_policy"):
            v = judge_task(t, d, "optimus", model=STOCK, task_problem_dir=problem)  # one retry
        if v.get("is_valid") is None and getattr(local, "err", None) == "bio_policy":
            v = judge_task(t, d, "optimus", model=FALLBACK, task_problem_dir=problem)
            if v.get("is_valid") in (True, False):
                v["judge_note"] = (f"The stock judge ({STOCK}) refused this task: OpenAI error code bio_policy, "
                                   "'This content was flagged for possible biological risk'. This verdict is from "
                                   f"{FALLBACK} via OpenRouter, using NatureBench's unchanged judge evidence, "
                                   "prompt, policy and parser.")
        jv.write_text(json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8")
        tag = {True: "VALID", False: "INVALID"}.get(v.get("is_valid"), "JUDGE ERROR")
        return f"{t}: {tag} [{v.get('model')}] {str(v.get('reason', ''))[:220]}"

    with ThreadPoolExecutor(max_workers=4) as ex:
        for line in ex.map(one, tasks):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
