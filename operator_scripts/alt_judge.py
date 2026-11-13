#!/usr/bin/env python3
"""Judge one task with a different model when the stock GPT-5.5 judge refuses it.

    python ~/alt_judge.py results/v11_full_gpu_low <task> [model]   (default anthropic/claude-sonnet-5.5)

Uses NatureBench's own judge_task: the same evidence collection, prompt, judging policy and verdict
parser. Only the model changes, called through OpenRouter. The verdict records the model and a
judge_note explaining why, for disclosure. Nothing is written unless a valid/invalid verdict comes back.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

NB = Path.home() / "NatureBench"
OR = "https://openrouter.ai/api"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    task = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "anthropic/claude-sonnet-5.5"
    tdir = NB / sys.argv[1] / task
    problem = NB / "data" / "naturebench_data" / "tasks" / task / "problem"
    key = (Path.home() / ".openrouter_key").read_text().strip()
    os.environ.update(JUDGE_MODEL=model, JUDGE_OPENAI_BASE_URL=OR, JUDGE_OPENAI_API_KEY=key,
                      JUDGE_ANTHROPIC_BASE_URL=OR, JUDGE_ANTHROPIC_API_KEY=key)
    os.chdir(NB)
    sys.path.insert(0, str(NB))
    import judge_core.clients as cl
    from judge import judge_task

    info: dict = {}

    def call(user_prompt, model=None, base_url=None, api_key=None, timeout=cl.DEFAULT_TIMEOUT):
        """NatureBench's judge request (same system prompt and user prompt), sent via OpenRouter."""
        body = {"model": model, "instructions": cl.JUDGE_SYSTEM, "input": user_prompt,
                "max_output_tokens": cl.MAX_JUDGE_OUTPUT_TOKENS}
        req = urllib.request.Request(cl.build_api_url(base_url, "/v1/responses"), data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
        info.clear()
        info["model"] = model
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            info["error"] = f"HTTP {e.code}: {detail}"
            return False, info["error"]
        except Exception as e:  # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {e}"
            return False, info["error"]
        out = payload.get("output", []) if isinstance(payload, dict) else []
        text = "".join(str(c.get("text", "")) for i in out if isinstance(i, dict)
                       for c in (i.get("content") or []) if isinstance(c, dict) and c.get("type") == "output_text")
        info.update(status=payload.get("status"), error_field=payload.get("error"),
                    incomplete=payload.get("incomplete_details"), text_chars=len(text))
        return True, text

    cl.call_openai_judge = call
    cl.call_anthropic_judge = call

    print(f"{task}: judging with {model} (NatureBench judge prompt and policy)...", flush=True)
    v = judge_task(task, tdir, "optimus", task_problem_dir=problem)
    for k, val in info.items():
        print(f"    {k}: {json.dumps(val)[:400]}")
    if v.get("is_valid") not in (True, False):
        print(f"\nNo verdict: {v.get('reason')}. Nothing written; paste this output.")
        return 1
    v["judge_note"] = ("The stock judge (openai/gpt-5.5) refused this task on every call: OpenAI error code "
                       "bio_policy, 'This content was flagged for possible biological risk'. This verdict is from "
                       f"{model} via OpenRouter, using NatureBench's unchanged judge evidence, prompt, policy and parser.")
    old = tdir / "judge_verdict.json"
    n = 1
    while (tdir / f"judge_verdict.error_{n}.json").exists():
        n += 1
    if old.is_file():
        (tdir / f"judge_verdict.error_{n}.json").write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
    old.write_text(json.dumps(v, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{task}: {'VALID' if v['is_valid'] else 'INVALID'} | {v.get('reason', '')[:600]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
